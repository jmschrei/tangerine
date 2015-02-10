# tangerine.py
# Contact: Jacob Schreiber
#          jmschr@cs.washington.edu

import sys, time, collections, hashlib, uuid, json
import itertools as it
from pomegranate import *

encrypt = lambda x: hashlib.sha512( x ).hexdigest()

def gen():
	for i in [1, 2, 3]:
		yield i

GENERATOR = gen()
PAGE_LIMIT = 4096
TYPE_NAMES = { 'str' : str,
			   'float' : float,
			   'int' : int }

class Page( object ):
	"""
	This represents a page of tuples. Pages can be read from into RAM, stored
	in cache, and written back out to disk.
	"""

	def __init__( self, tuples=[], id=None ):
		"""
		Input the tuples to be stored together.
		"""

		self.id = id or uuid.uuid4().hex
		self.tuples = tuples
		self.n = len( self.tuples )

	def __str__( self ):
		"""
		Return the string representation of the page.
		"""

		return "\n".join( "\t".join( map( str, tup ) ) for tup in self.tuples )

	def insert( self, tuple ):
		"""
		Insert a tuple into the page.
		"""

		for i in xrange( len(self.tuples) ):
			if self.tuples[i] is None:
				self.tuples[i] = tuple
		else: 
			self.tuples.append( tuple )

		self.n += 1

	def delete( self, filter_expr ):
		"""
		Take in a dictionary of keys and the value that should be filtered
		out. Must be all of them.
		"""

		for i in xrange( self.n ):
			for key, value in filter_expr.items():
				if self.tuples[i][key] != value:
					break
			else:
				self.tuples[i] = None
				self.n -= 1

	def drop( self ):
		"""
		Drop this page.
		"""

		os.remove( self.id + '.page' )
		del self

	def scan( self ):
		"""
		Return tuples one at a time.
		"""

		for tup in self.tuples:
			yield tup

	def commit( self ):
		"""
		Commit the page to memory.
		"""

		data = { 'n' : self.n,
				 'id' : self.id,
				 'tuples' : self.tuples }

		with open( self.id+'.page', 'w' ) as outfile:
			outfile.write( json.dumps( data,
									   indent=4,
									   separators=(',', ' : ')))

	@classmethod
	def load( cls, filename ):
		"""
		Load a page from memory.
		"""

		assert filename.endswith('.page'), "Not a valid page file"
		with open( filename, 'r' ) as infile:
			data = ''.join( line.strip('\r\n') for line in infile )

		data = json.loads( data )
		data['tuples'] = map( tuple, data['tuples'])

		return cls( data['tuples'], data['id'] )

class Table( object ):
	"""
	This represents a table in a database. It should store the names and types
	of each column, in addition to the data. It will do this as a list of
	tuples.
	"""

	def __init__( self, name, column_names, types, data=None, page_ids=[], pagestack_ids=[] ):
		"""
		Input the name of the table, name of the columns, types of the columns,
		and the pages. 
		"""

		self.name = name
		self.column_names = column_names
		self.column_name_map = {name: i for i, name in enumerate(column_names)}
		self.types = types
		self.display_limit = 20
		
		self.page_ids = page_ids
		self.pagestack_ids = pagestack_ids
		if len( page_ids ) > 0:
			self.pages = [ Page.load( i+'.page' ) for i in page_ids ]
		else:
			self.pages = []

		self.pagestack = [ page for page in self.pages if page.id in pagestack_ids ]

		self.data = data
		if type(self.data) == list and len(self.data) > 0:
			self.insert( self.data )

	def __str__( self ):
		"""
		Return a string representation of the data.
		"""

		width = max( max( map( len, self.column_names ) ), 9 )


		header = '| '.join( '{1:{0}} {2:3} '.format( width, cn, ct.__name__ ) 
			for cn, ct in it.izip( self.column_names, self.types ) )

		return header + '\n' + '-'*len(header) + '\n' + \
			'\n'.join( '| '.join( '{1:{0}} '.format( width+4, str(t) ) for t in tup ) for i, tup in enumerate(self.scan()) if i < self.display_limit ) + '\n'

	def scan( self ):
		"""
		Loading the data a bit a few pages at a time from memory.
		"""

		# If we have a generator representing operations on previous data, as
		# opposed to raw data, then go through that generator one at a time.
		if type(self.data) == type(GENERATOR):
			for tup in self.data:
				yield tup

		# If we have raw data stored on pages, go through the pages one tuple
		# at a time.
		else:
			for page in self.pages:
				for tup in page.scan():
					yield tup

	def insert( self, data ):
		"""
		Insert tuples into the table. If a single tuple, then just insert that.
		If a list of tuples, then insert all of the tuples into the table.
		"""

		# If inserting multiple tuples, recursively add them one at a time
		# to reduce the amount of code I need to write.
		if type(data) == list:
			self.insert( data[0] )
			if len(data) > 1:
				self.insert( data[1:] )
			return

		# If no unfilled pages in the table, make a new page.
		if len( self.pagestack ) == 0:
			page_id = uuid.uuid4().hex
			page = Page( id=page_id )
			self.page_ids.append( page_id )
			self.pagestack_ids.append( page_id )
			self.pagestack.append( page )
			self.pages.append( page )

		# Otherwise select the top page.
		else:
			page = self.pagestack[-1]

		# Ensure the tuple is a valid tuple to be inserted
		for i, ct in it.izip( data, self.types ):
			if type(i) != ct:
				print "INSERTION ERROR: Cannot insert data of type" +\
					" {} into column of type {}".format( type(i).__name__, ct.__name__ )
				break
		else:
			page.insert( data )

		# If the page is now full, remove it.
		if sys.getsizeof( page.tuples ) > PAGE_LIMIT:
			self.pagestack.pop()
			self.pagestack_ids.pop()

	def drop( self ):
		"""
		Drop this table by dropping all the pages. 
		"""

		for page in self.pages:
			page.drop()

		os.remove( self.name + '.table' )

	def _block_join( self, other, self_on, other_on ):
		"""
		Perform basic block join. 
		"""

		tcolumns = map( self.column_name_map.__getitem__, self_on ) if self_on else []
		ocolumns = map( other.column_name_map.__getitem__, other_on ) if other_on else []

		# For each tuple in the other table
		for x in self.scan():
			# For each tuple in the other table
			for y in other.scan():
				# For each column we are joining on
				for tcolumn, ocolumn in it.izip( tcolumns, ocolumns ):
					# Make sure that the columns match between tuples
					if x[tcolumn] != y[ocolumn]:
						break
				else:
					yield x + y

	def _hash_join( self, other, self_on, other_on ):
		"""
		Perform a hash join, where this table is the one the hash map is
		made on.
		"""

		hash_map = {}
		tcolumns = map( self.column_name_map.__getitem__, self_on ) if self_on else []
		ocolumns = map( other.column_name_map.__getitem__, other_on ) if other_on else []

		# Build a hash map on the self table
		hash_col = tcolumns[0]
		for tup in self.scan():
			val = hash_map[ tup[hash_col] ]
			
			if val in hash_map.keys():
				hash_map[val].append( tup )
			else:
				hash_map[val] = [ tup ]

		# Run all the tuples from the other table through the hash map
		other_col = ocolumns[0]
		for y in other.scan():
			self_tups = hash_map[ tup[other_col] ]

			# Go through all tuples which match
			for x in self_tups:
				for tcolumn, ocolumn in it.izip( tcolumns[1:], ocolumns[1:] ):
					if x[tcolumn] != y[ocolumn]:
						break
				else:
					yield x + y 

	def join( self, other_table, self_on=None, other_on=None, algorithm="block" ):
		"""
		Perform a table join on two tables. Specify the names of the columns
		for the join to be on.
		"""

		this_cns = self.column_names
		other_cns = other_table.column_names
		new_table_names = [ "{}{}{}".format( self.name, '.' if self.name != '' else '', cn ) for cn in this_cns ] + \
			[ "{}.{}".format( other_table.name, cn ) for cn in other_cns ]
		
		new_table_types = self.types + other_table.types


		join = self._block_join if algorithm == 'block' or not self_on else self._hash_join

		return Table( "", 
					  new_table_names, 
					  new_table_types, 
					  join( other_table, self_on, other_on ) )

	def _projection( self, on ):
		"""
		Internal generator yielding tuples one at a time.
		"""

		indices = map( self.column_name_map.__getitem__, on )

		for tup in self.scan():
			yield tuple( tup[i] for i in indices )

	def projection( self, on ):
		"""
		Return the table object removing some of the columns.
		"""

		if on == '*' or on == ['*']:
			return self

		indices = map( self.column_name_map.__getitem__, on )
		column_names = [ self.column_names[i] for i in indices ]
		types = [ self.types[i] for i in indices ]

		return Table( "",
					  column_names,
					  types,
					  self._projection( on ) )
	
	def _selection( self, filter_expr ):
		"""
		Internal generator yielding tuples one at a time.
		"""

		for tup in self.scan():
			if filter_expr( tup, self.column_name_map ):
				yield tup

	def selection( self, filter_expr ):
		"""
		Return the table object removing some of the tuples.
		"""

		return Table( "",
					  self.column_names,
					  self.types,
					  self._selection( filter_expr ) ) 

	def groupby( self, attribute, aggregate, aggregate_attribute=None ):
		"""
		Group by a particular attribute, given some aggregate. Aggregates include:
			'MIN', 'MAX, 'SUM', 'AVG', 'COUNT'
		"""

		t = float if aggregate != 'COUNT' else int
		summary = collections.defaultdict(t)
		attr_column = self.column_name_map[attribute]

		aggregate_attribute = aggregate_attribute or attribute
		agg_column = self.column_name_map[aggregate_attribute]

		n = 0.
		for tup in self.scan():
			attr_val = tup[attr_column]
			agg_val = tup[agg_column]

			if aggregate == 'COUNT':
				summary[attr_val] += 1
			elif aggregate == 'MIN':
				if summary[attr_val] > agg_val:
					summary[attr_val] = agg_val
			elif aggregate == 'MAX':
				if summary[attr_val] < agg_val:
					summary[attr_val] = agg_val
			elif aggregate == 'SUM' or aggregate == 'MEAN':
				summary[attr_val] += agg_val

		if aggregate == 'MEAN':
			for key, val in summary.items():
				summary[key] = val / n 

		column_names = [ attribute, "{}({})".format( aggregate, aggregate_attribute ) ]
		data = [ ( a, b ) for a, b in summary.items() ]

		return Table( "",
					  column_names,
					  [ self.types[attr_column], t ],
					  data )

	def commit( self ):
		"""
		Commit the data to memory.
		"""

		for page in self.pages:
			page.commit()

		data = { 
				 'name' : self.name,
				 'column_names' : self.column_names,
				 'types' : [ t.__name__ for t in self.types ],
				 'page_ids' : self.page_ids,
				 'pagestack_ids' : self.pagestack_ids,
			   }

		with open( self.name+'.table', 'w' ) as outfile:
			outfile.write( json.dumps( data,
									   indent=4,
									   separators=(',', ' : ')))

	@classmethod
	def load( cls, filename ):
		"""
		Load from a table metainformation file.
		"""

		assert filename.endswith( '.table' )
		with open( filename, 'r' ) as infile:
			data = ''.join( line.strip('\r\n') for line in infile )

		data = json.loads( data )
		data['types'] = [ TYPE_NAMES[t] for t in data['types'] ]

		return cls( data['name'], data['column_names'], data['types'], 
					page_ids=data['page_ids'], pagestack_ids=data['pagestack_ids'] )

	def to_csv( self, filename ):
		"""
		Write the data to a CSV file with headers indicating the name and type.
		"""

		names = self.column_names
		types = self.types

		with open( filename, 'w' ) as outfile:
			# Make the headers which is a '{name} {type}' pair for each column
			headers = ( "{} {}".format( n, t ) for n, t in zip( names, types ) )

			# Write the headers out
			outfile.write( ",".join( headers ) + "\n" )

			# Now write out each tuple
			for tup in self.data:
				outfile.write( ",".join( map( str, tup ) ) + "\n" )

	@classmethod
	def from_csv( cls, filename ):
		"""
		Open a csv file with headers indicating the name and the type.
		"""

		# Get the name of the table from the filename
		name = filename.split('\\')[-1].split('.')[0]

		# Open the csv file
		with open( filename, 'r' ) as infile:
			# Get the names and types from 
			header = infile.readline().strip("\r\n").split(',')
			header = [ title.split() for title in header ]

			names = tuple([ x[0] for x in header ])
			types = tuple([ TYPE_NAMES[x[1]] for x in header ])
			data = []

			for l in infile:
				l = l.strip("\r\n").split(',')
				l = [ cast(item) for cast, item in zip( types, l ) ]
				data.append( tuple(l) )

		return cls( name, names, types, data )

class User( object ):
	"""
	A user which can access this database.
	"""

	def __init__( self, username, password ):
		self.username = username
		self.password = password

	def __str__( self ):
		'''
		JSON string representation of a user.
		'''

		return json.dumps( { 'username' : self.username,
							 'password' : self.password },
							 indent=4,
							 separators=(',', ' : ') )

	def __repr__( self ):
		return json.dumps( { 'username' : self.username,
							 'password' : self.password },
							 indent=4,
							 separators=(',', ' : ') )

class Database( object ):
	"""
	A database object, stores tables and must be 'connected' to. 
	"""

	def __init__( self, name="", users=[], table_names=[] ):
		self.name = name
		self.users = users 
		self.table_names = table_names
		self.table_map = {}
		self.tables = []
		self.fsm = SQLFSM()

	def add_user( self, username, password ):
		"""
		Add a user which can access this database.
		"""

		user = User( username=encrypt(username),
					 password=encrypt(password))
		self.users.append( user )

	def add_table( self, name, column_names, types ):
		"""
		Add a table to the database.
		"""

		if name in self.table_map.keys():
			raise Warning( "Table {} already in database.".format( name ) )
		else:
			table = Table( name, column_names, types )
			self.table_map[name] = table
			self.table_names.append( name )
			self.tables.append( table )

	def drop_table( self, name ):
		"""
		Drop a table from the database, removing all its pages as well.
		"""

		# Call the tables drop method, deleting it and its pages
		self.table_map[name].drop()

		# Now delete all the metadata associated with that table from the DB
		self.table_names.remove( name )
		self.tables.remove( self.table_map[name] )
		del self.table_map[name]
		self.commit()

	def connect( self, username, password ):
		"""
		Attempt to connect to the database.
		"""

		for user in self.users:
			if encrypt(username) == user.username and encrypt(password) == user.password:
				self.tables = [ Table.load( name+'.table' ) for name in self.table_names ]
				self.table_map = { name : table for name, table in zip( self.table_names, self.tables ) }
				break
		else:
			raise Warning( "Username password/combination does not work." )

	def execute( self, query ):
		"""
		Implement a lite SQL command. Parser is barely functional.
		"""

		fsm = self.fsm
		
		# Process the query a bit
		for item in 'GROUP BY', 'CREATE TABLE', 'INSERT INTO', 'DROP TABLE', 'HASH JOIN':
			query = query.replace( item, item.replace(' ', '') )
		query = query.split()

		# Assign tags to each of the states
		try:
			tags = [ state.name for i, state in fsm.viterbi( query )[1] if not state.is_silent() ]
		except:
			return "INVALID QUERY"

		# If commiting the database...
		if tags[0] == 'commit':
			self.commit()
			return "DATABASE COMMIT"

		# Handle insertions into the database gracefully.
		elif tags[0] == 'insertinto':
			table_name = None
			values = []

			str_values = ''
			for q, tag in zip( query, tags ):
				if tag == 'nkt':
					table_name = q
				elif tag == 'nkv':
					str_values += q.replace('"', '').replace("'", "") + ' '
			
			str_values = str_values.split(',')
			for v in str_values:
				try:
					values.append( int(v) )
				except:
					try:
						values.append( float(v) )
					except:
						values.append(v.strip(' '))

			self.insert( table_name, tuple(values) )
			return "DATABASE INSERT"

		# If we're creating a table, we need to get the names of the columns
		# and the types of those columns, as well as the name of the database
		elif tags[0] == 'createtable':
			table_name = None

			values = ''
			for q, tag in zip( query, tags ):
				if tag == 'nkt':
					table_name = q
				elif tag == 'nkv':
					values += q + ' '

			# Get one long string which is all column names and types
			values = values.replace(',', '').split()
			column_names = values[::2]
			types = map( TYPE_NAMES.__getitem__, values[1::2] )

 			# Use the internal method to add the table.
			self.add_table( table_name, column_names, types )
			return "ADD TABLE"

		elif tags[0] == 'droptable':
			for q, tag in zip( query, tags ):
				if tag == 'nkt':
					table_name = q

			self.drop_table( table_name )
			return "DROP TABLE"

		# Go through the tags.
		elif tags[0] == 'select':
			# Initiate variables to store query data
			column_names = []
			table_names = []
			where_clauses = []
			groupby = None
			limit = 20
			join_attrs = []
			join_type = 'block'

			# Pull the information from the tagged query
			for q, tag in zip( query, tags ):
				if tag == 'nks':
					column_names.append( q.replace(',', '') )
				elif tag == 'nkf':
					table_names.append( q.replace(',', '') )
				elif tag == 'nkw':
					where_clauses.append( q.replace(',', '') )
				elif tag == 'nkgb':
					groupby = q
				elif tag == 'nkl':
					limit = q
				elif tag == 'hashjoin':
					join_type = 'hash'

			tables = map( self.table_map.__getitem__, table_names )

			# Now join all the tables together. At first this may sound
			# inefficient, but remember that nothing is being calculated,
			# it's just a generator which is being built
			t = tables[0]
			for table in tables[1:]:
				t = t.join( table, algorithm=join_type )

			# Now convert the selection criteria into lambda expressions which
			# can be evaluated by the table
			for i, clause in enumerate( where_clauses ):
				split_clause = None
				for char in '>=', '<=', '=', '<', '>':
					if char in clause:
						split_clause = clause.split( char )
						split_clause = [ split_clause[0], char, split_clause[1] ]
						break

				if len( table_names ) == 1:
					table = self.table_map[table_names[0]]
				else:
					table = self.table_map[split_clause[0].split('.')[0]]

				split_clause[0] = 'x[y["{}"]]'.format( split_clause[0] )
				split_clause[1] = '==' if split_clause[1] == '=' else split_clause[1] 

				try:
					int(split_clause[2])
					func = eval( 'lambda x, y: {} {} {}'.format( *split_clause ) )
					t = t.selection( func )
				except:
					split_clause[2] = 'x[y["{}"]]'.format( split_clause[2] )
					func = eval( 'lambda x, y: {} {} {}'.format( *split_clause ) )
					t = t.selection( func )

			t.display_limit = limit
			return t.projection( column_names )

	def insert( self, table, tuples ):
		"""
		Insert tuples into the specified table.
		"""

		self.table_map[table].insert( tuples )

	def close( self ):
		"""
		Close this database connection.
		"""

		self.commit()
		del self

	def commit( self ):
		"""
		Commit the database to a file. Commits all tables as well.
		"""

		for table in self.tables:
			table.commit()

		db_json = json.dumps( { 'name' : self.name,
								'users' : [ user.__dict__ for user in self.users ],
								'table_names' : self.table_names
							  },
							  indent=4,
							  separators = (',', ' : ') )

		with open( self.name+'.db', 'w' ) as outfile:
			outfile.write( db_json )

	@classmethod
	def load( cls, filename ):
		"""
		Load a database object from the file.
		"""

		assert filename.endswith('.db'), "Not a valid database file."
		with open( filename, 'r' ) as infile:
			db_json = ''.join( line.strip('\r\n') for line in infile )
		
		db = json.loads( db_json )
		db['users'] = [ User( d['username'], d['password'] ) for d in db['users'] ]
		return cls( db['name'], db['users'], db['table_names'] )

def SQLFSM():
	"""
	Create and return a simple FSM for tagging parts of the query. I use a HMM
	object, but the parameters I feed in make it a FSM.
	"""

	keywords = ( 'SELECT', 'GROUPBY', 'LIMIT', 'FROM', 'WHERE', 'CREATETABLE',
				 'CREATEVIEW', 'VALUES', 'INSERTINTO', '(', ')', 'AS', 'AND',
				 'CONNECTTO', 'DROPTABLE', 'HASHJOIN' ) 
	not_keyword = lambda x: 0 if x not in keywords else float("-inf") 
	model = HiddenMarkovModel( "SQLParser" )

	non_keyword_select   = State( LambdaDistribution( not_keyword ), name='nks' )
	non_keyword_from     = State( LambdaDistribution( not_keyword ), name='nkf' )
	non_keyword_where    = State( LambdaDistribution( not_keyword ), name='nkw' )
	non_keyword_groupby  = State( LambdaDistribution( not_keyword ), name='nkgb' )
	non_keyword_limit    = State( LambdaDistribution( not_keyword ), name='nkl' )
	non_keyword_table    = State( LambdaDistribution( not_keyword ), name='nkt' )
	non_keyword_table_cn = State( LambdaDistribution( not_keyword ), name='nkcn' )
	non_keyword_table_ct = State( LambdaDistribution( not_keyword ), name='nkct' )
	non_keyword_values   = State( LambdaDistribution( not_keyword ), name='nkv' )

	select       = State( DiscreteDistribution( {'SELECT' : 1.0} ), name='select' )
	froms        = State( DiscreteDistribution( {'FROM' : 1.0 } ), name='from' )
	where        = State( DiscreteDistribution( {'WHERE' : 1.0 } ), name='where' )
	groupby      = State( DiscreteDistribution( {'GROUPBY' : 1.0 } ), name='groupby' )
	limit        = State( DiscreteDistribution( {'LIMIT' : 1.0 } ), name='limit' )
	insertinto   = State( DiscreteDistribution( {'INSERTINTO' : 1.0 } ), name='insertinto' )
	createtable  = State( DiscreteDistribution( {'CREATETABLE' : 1.0 } ), name='createtable' )
	values       = State( DiscreteDistribution( {'VALUES' : 1.0 } ), name='values' )
	left_parens  = State( DiscreteDistribution( {'(' : 1.0 } ), name='(' )
	right_parens = State( DiscreteDistribution( {')' : 1.0 } ), name=')' )
	and_state    = State( DiscreteDistribution( {'AND' : 1.0 } ), name='and' )
	commit       = State( DiscreteDistribution( {'COMMIT' : 1.0 } ), name='commit' )
	drop         = State( DiscreteDistribution( {'DROPTABLE' : 1.0 } ), name='droptable' )
	hashjoin     = State( DiscreteDistribution( {'HASHJOIN' : 1.0} ), name='hashjoin' )

	model.add_states([ non_keyword_select, non_keyword_from,    non_keyword_table,
					   non_keyword_where,  non_keyword_groupby, non_keyword_limit,
					   non_keyword_table,  non_keyword_values,
					   select, froms, where, groupby, limit, 
					   insertinto, createtable, values, left_parens, right_parens,
					   and_state, commit, drop, hashjoin ])

	model.add_transition( model.start, select, 0.166 )
	model.add_transition( select, non_keyword_select, 1.0 )
	model.add_transition( non_keyword_select, non_keyword_select, 0.5 )
	model.add_transition( non_keyword_select, froms, 0.5 )
	
	model.add_transition( froms, non_keyword_from, 1.0 )
	model.add_transition( non_keyword_from, non_keyword_from, 0.166 )
	model.add_transition( non_keyword_from, where, 0.166 )
	model.add_transition( non_keyword_from, groupby, 0.166 )
	model.add_transition( non_keyword_from, limit, 0.166 )
	model.add_transition( non_keyword_from, model.end, 0.166 )
	model.add_transition( non_keyword_from, hashjoin, 0.17 )

	model.add_transition( hashjoin, where, 0.25 )
	model.add_transition( hashjoin, groupby, 0.25 )
	model.add_transition( hashjoin, limit, 0.25 )
	model.add_transition( hashjoin, model.end, 0.25 )
	
	model.add_transition( where, non_keyword_where, 1.0 )
	model.add_transition( non_keyword_where, non_keyword_where, 0.20 )
	model.add_transition( non_keyword_where, groupby, 0.20 )
	model.add_transition( non_keyword_where, limit, 0.20 )
	model.add_transition( non_keyword_where, model.end, 0.20 )
	model.add_transition( non_keyword_where, and_state, 0.20 )
	model.add_transition( and_state, non_keyword_where, 1.0 )

	model.add_transition( groupby, non_keyword_groupby, 1.0 )
	model.add_transition( non_keyword_groupby, limit, 0.5 )
	model.add_transition( non_keyword_groupby, model.end, 0.5 )

	model.add_transition( limit, non_keyword_limit, 1.0)
	model.add_transition( non_keyword_limit, model.end, 1.0 )

	model.add_transition( model.start, insertinto, 0.166 )
	model.add_transition( insertinto, non_keyword_table, 1.0 )
	model.add_transition( non_keyword_table, left_parens, 0.33 )
	model.add_transition( non_keyword_table, values, 0.33 )
	model.add_transition( non_keyword_table, model.end, 0.34 )
	model.add_transition( values, left_parens, 0.5 )
	model.add_transition( left_parens, non_keyword_values, 1.0 )
	model.add_transition( non_keyword_values, non_keyword_values, 0.5 )
	model.add_transition( non_keyword_values, right_parens, 0.5 )
	model.add_transition( right_parens, model.end, 1.0 )

	model.add_transition( model.start, createtable, 0.166 )
	model.add_transition( createtable, non_keyword_table, 1.0 )

	model.add_transition( model.start, commit, 0.166 )
	model.add_transition( commit, model.end, 1.0 )

	model.add_transition( model.start, drop, 1.7 )
	model.add_transition( drop, non_keyword_table, 1.0 )

	model.bake() 
	return model

def connect( database, username, password ):
	"""
	Connect to a database, given a username and password.
	"""

	db = Database.load( database )

	try:
		db.connect( username, password )
	except:
		raise Warning( "Incorrect username/password combination." )
	else:
		return db

if __name__ == '__main__':
	# If this is called directly, create a terminal.
	_, db_name, user, pwd = sys.argv
	db = connect( db_name, user, pwd )

	# Try to connect to the database
	try:
		db = connect( db_name, user, pwd )
	except:
		print "FATAL ERROR: Incorrect username or password or database."
		sys.exit()

	db_name = db_name.strip('.db')

	# Do command line stuff
	while True:
		query = ''
		i = True
		while not query.endswith( '; ' ):
			query += raw_input( "{}{}# ".format( db_name, '=' if i else '-' ) ).strip(' ') + ' '
			i = False

		if query.lower() in ('exit; ', 'quit; '):
			break

		print db.execute( query[:-2] )

		'''
		try:
			result = db.execute( query[:-2] )
			print result
		except Exception:
			print "INVALID QUERY"
		'''