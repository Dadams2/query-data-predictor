## 0x0a import into postgresql 

```sql
self = <query_data_predictor.sdss_importer.DataLoader object at 0x1053e63b0>

    def _load_data(self) -> None:
        """Load JSON data into PostgreSQL."""
        if not os.path.exists(self.json_file_path):
            raise FileNotFoundError(f"JSON file not found: {self.json_file_path}")
    
        if not self.conn:
            raise ConnectionError("Database connection not established")
    
        try:
            # Create a cursor
            with self.conn.cursor() as cur:
                # Create table if it doesn't exist
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS query_data (
                        data JSONB
                    )
                """)
    
                # Check if table is empty before loading data
                cur.execute("SELECT COUNT(*) FROM query_data")
                count = cur.fetchone()[0]
    
                if count == 0:
                    # Use PostgreSQL's COPY command to efficiently load data
                    with open(self.json_file_path, 'r') as f:
                        # For JSON files with one object per line
                        cur.execute("""
                            CREATE TEMP TABLE temp_json (data TEXT);
                        """)
    
                        # Copy the file content to the temporary table
                        cur.copy_from(f, 'temp_json', columns=('data',))
    
                        # Insert from temp table to main table as JSONB
                        cur.execute("""
                            INSERT INTO query_data
                            SELECT data::JSONB FROM temp_json;
                        """)
    
                        # Clean up
                        cur.execute("DROP TABLE temp_json")
    
                # Create a view to access JSON attributes directly
                cur.execute(f"""
                    CREATE OR REPLACE VIEW query_data_view AS
                    SELECT
                        data->>'sessionID' as {SESSION_ID_COLUMN},
                        data->>'statementID' as {STATEMENT_ID_COLUMN},
                        data
                    FROM query_data
                """)
    
                self.conn.commit()
    
        except psycopg2.Error as e:
            self.conn.rollback()
>           raise Exception(f"Error loading data into PostgreSQL: {e}")
E           Exception: Error loading data into PostgreSQL: invalid input syntax for type json
E           DETAIL:  Character with value 0x0a must be escaped.
E           CONTEXT:  JSON data, line 1: ...,"999":53228078},"statement":{"0":"SELECT * from
```

This error typically occurs in PostgreSQL when you're trying to insert or update data containing unescaped newline (line feed) characters, represented as 0x0a in hexadecimal.

Here are some ways to diagnose and fix this:

Identify the problematic data:

Look for text values containing newlines in your data
Check if you're constructing queries by concatenating strings
Fix the issue:

Use parameterized queries instead of string concatenation:
Or escape newlines in your data before inserting:
If using bulk operations like COPY or \copy:

Make sure your data format properly escapes special characters
Consider using a library that handles the escaping for you
Without seeing your specific code, these general approaches should help address the issue.


## Can't use psycopg2 to write tables to a database from pandas

```
        except Exception as e:
            if self.conn:
                self.conn.rollback()
>           raise Exception(f"Error loading data into PostgreSQL: {e}")
E           Exception: Error loading data into PostgreSQL: syntax error at end of input
E           LINE 3:                 
E                                   ^

src/query_data_predictor/sdss_json_importer.py:118: Exception
```

https://duckduckgo.com/?q=convertread+pandas+dataframe+into+postgresql+using+psycopg2&t=newext&atb=v418-1&ia=web

https://stackoverflow.com/questions/61280382/writing-dataframe-to-postgres-database-psycopg2

https://stackoverflow.com/questions/23103962/how-to-write-dataframe-to-postgres-table

you have to use sqlalchemy

## Error executing current query 

Error executing current query: current transaction is aborted, commands ignored until end of transaction block

https://stackoverflow.com/questions/10399727/psqlexception-current-transaction-is-aborted-commands-ignored-until-end-of-tra

## __file__ does not exist in jupyter notebooks 

https://stackoverflow.com/questions/39125532/file-does-not-exist-in-jupyter-notebook

```
DATA_PATH = Path(__file__).parent / "data" / "sdss_joined_sample.json"

NameError                                 Traceback (most recent call last)
Cell In[2], line 1
----> 1 DATA_PATH = Path(__file__).parent / "data" / "sdss_joined_sample.json"

NameError: name '__file__' is not defined
```

## Polars gotchas

polars is a pretty good substitute for pandas but sometimes it cannot be helped
```
        if f"{type(df)}" == "<class 'pandas.core.frame.SparseDataFrame'>":
            msg = (
                "SparseDataFrame support has been deprecated in pandas 1.0,"
                " and is no longer supported in mlxtend. "
                " Please"
                " see the pandas migration guide at"
                " https://pandas.pydata.org/pandas-docs/"
                "stable/user_guide/sparse.html#sparse-data-structures"
                " for supporting sparse data in DataFrames."
            )
            raise TypeError(msg)
    
>       if df.size == 0:
E       AttributeError: 'DataFrame' object has no attribute 'size'
```

## Seralising Memory view objects

```
Dataset for session 11023 saved to /Users/DAADAMS/Other/query-data-predictor/data/datasets/query_prediction_session_11023.pkl with 9 samples
Processing session 11024 with 4066 queries
Error executing current query: cannot pickle 'memoryview' object
Error executing current query: cannot pickle 'memoryview' object
Error executing current query: cannot pickle 'memoryview' object
Error executing current query: cannot pickle 'memoryview' object
Error executing current query: cannot pickle 'memoryview' object
Error executing current query: cannot pickle 'memoryview' object
Error executing current query: cannot pickle 'memoryview' object
Error executing current query: cannot pickle 'memoryview' object
Error executing current query: cannot pickle 'memoryview' object
Error executing current query: cannot pickle 'memoryview' object
Error executing current query: cannot pickle 'memoryview' object
Error executing current query: cannot pickle 'memoryview' object
Error executing current query: cannot pickle 'memoryview' object
Error executing current query: cannot pickle 'memoryview' object
Error executing current query: cannot pickle 'memoryview' object
Error executing current query: cannot pickle 'memoryview' object
Error executing current query: cannot pickle 'memoryview' object
^C

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! KeyboardInterrupt !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
/Users/DAADAMS/.local/share/uv/python/cpython-3.10.15-macos-aarch64-none/lib/python3.10/encodings/utf_8.py:15: KeyboardInterrupt
(to show a full traceback on KeyboardInterrupt use --full-trace)
==================================================== no tests ran in 40.45s ====================================================
❯ 
❯ uv add ipython
Resolved 73 packages in 520ms
      Built query-data-predictor @ file:///Users/DAADAMS/Other/query-data-predictor
Prepared 1 package in 244ms
Uninstalled 1 package in 1ms
Installed 1 package in 2ms
 ~ query-data-predictor==0.1.0 (from file:///Users/DAADAMS/Other/query-data-predictor)
❯ 
❯ 
❯ uv add --dev ipython
Resolved 73 packages in 16ms
      Built query-data-predictor @ file:///Users/DAADAMS/Other/query-data-predictor
Prepared 1 package in 169ms
Uninstalled 1 package in 0.52ms
Installed 1 package in 1ms
 ~ query-data-predictor==0.1.0 (from file:///Users/DAADAMS/Other/query-data-predictor)
❯ pytest -vs experiments/build_csv_dataset.py
===================================================== test session starts ======================================================
platform darwin -- Python 3.10.15, pytest-8.3.5, pluggy-1.5.0 -- /Users/DAADAMS/Other/query-data-predictor/.venv/bin/python
cachedir: .pytest_cache
rootdir: /Users/DAADAMS/Other/query-data-predictor
configfile: pyproject.toml
plugins: cov-6.1.1
collected 1 item                                                                                                               

experiments/build_csv_dataset.py::test_full_build Processing session 11015 with 19 queries
Dataset for session 11015 saved to /Users/DAADAMS/Other/query-data-predictor/data/datasets/query_prediction_session_11015.pkl with 19 samples
Processing session 11016 with 31 queries
Dataset for session 11016 saved to /Users/DAADAMS/Other/query-data-predictor/data/datasets/query_prediction_session_11016.pkl with 31 samples
Processing session 11017 with 13 queries
Dataset for session 11017 saved to /Users/DAADAMS/Other/query-data-predictor/data/datasets/query_prediction_session_11017.pkl with 13 samples
Processing session 11018 with 10 queries
Dataset for session 11018 saved to /Users/DAADAMS/Other/query-data-predictor/data/datasets/query_prediction_session_11018.pkl with 10 samples
Processing session 11019 with 4 queries
Dataset for session 11019 saved to /Users/DAADAMS/Other/query-data-predictor/data/datasets/query_prediction_session_11019.pkl with 4 samples
Processing session 11022 with 44 queries
Python 3.10.15 (main, Oct 16 2024, 08:33:15) [Clang 18.1.8 ]
Type 'copyright', 'credits' or 'license' for more information
IPython 8.36.0 -- An enhanced Interactive Python. Type '?' for help.

In [1]: e
Out[1]: TypeError("cannot pickle 'memoryview' object")

In [2]: e.with_traceback
Out[2]: <function TypeError.with_traceback>

In [3]: e.with_traceback()
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
File ~/Other/query-data-predictor/src/query_data_predictor/dataset_creator.py:207, in DatasetCreator.build_dataset(self, session_id)
    206     with open(results_filepath, 'wb') as f:
--> 207         pickle.dump(curr_results, f)
    209 except Exception as e:

TypeError: cannot pickle 'memoryview' object

During handling of the above exception, another exception occurred:

TypeError                                 Traceback (most recent call last)
Cell In[3], line 1
----> 1 e.with_traceback()

TypeError: BaseException.with_traceback() takes exactly one argument (0 given)

In [4]: import traceback

In [5]: traceback.print_exc()
Traceback (most recent call last):
  File "/Users/DAADAMS/Other/query-data-predictor/src/query_data_predictor/dataset_creator.py", line 207, in build_dataset
    pickle.dump(curr_results, f)
TypeError: cannot pickle 'memoryview' object

In [6]: curr_results
Out[6]: 
              fieldid  zoom  run  rerun  ...        cy        cz           htmid                                   img
0  588848900446814208     0  756     44  ...  0.108531  0.005491  15403055903110  [b'0', b'x', b'1', b'1', b'1', b'1']
1  588848900446814208     5  756     44  ...  0.108531  0.005491  15403055903110  [b'0', b'x', b'1', b'1', b'1', b'1']
2  588848900446814208    10  756     44  ...  0.108531  0.005491  15403055903110  [b'0', b'x', b'1', b'1', b'1', b'1']
3  588848900446814208    15  756     44  ...  0.108531  0.005491  15403055903110  [b'0', b'x', b'1', b'1', b'1', b'1']
4  588848900446814208    20  756     44  ...  0.108531  0.005491  15403055903110  [b'0', b'x', b'1', b'1', b'1', b'1']
5  588848900446814208    25  756     44  ...  0.108531  0.005491  15403055903110  [b'0', b'x', b'1', b'1', b'1', b'1']
6  588848900446814208    30  756     44  ...  0.108531  0.005491  15403055903110  [b'0', b'x', b'1', b'1', b'1', b'1']

[7 rows x 29 columns]

In [7]: type(curr_results)
Out[7]: pandas.core.frame.DataFrame

In [8]: type(curr_results['img'][0])
Out[8]: memoryview

In [9]: current_query
Out[9]: "select * from Frame where fieldId=x'082c02f481830000'::bigint"

In [10]: [23;9~
```

https://stackoverflow.com/questions/16351286/python-query-objects-are-not-serializable

## multiple columns withmae name pandas 

so if you have a query that returns two columns with the same name but different values you don't get the dtype accesor

```
In [3]: import traceback

In [4]: traceback.print_exc()
Traceback (most recent call last):
  File "/Users/DAADAMS/Other/query-data-predictor/src/query_data_predictor/dataset_creator.py", line 222, in build_dataset
    result_features = self._extract_result_features(curr_columns, curr_results)
  File "/Users/DAADAMS/Other/query-data-predictor/src/query_data_predictor/dataset_creator.py", line 114, in _extract_result_features
    col_type = results[col].dtype
  File "/Users/DAADAMS/Other/query-data-predictor/.venv/lib/python3.10/site-packages/pandas/core/generic.py", line 6299, in __getattr__
    return object.__getattribute__(self, name)
AttributeError: 'DataFrame' object has no attribute 'dtype'. Did you mean: 'dtypes'?
```

```
In [20]: results['z']
Out[20]: 
            z         z
0   15.344472  0.079881
1   18.994343  2.032640
2   15.895847  0.108700
3   16.803057  0.113331
4   15.514304  0.065348
..        ...       ...
95  16.898823  0.000091
96  14.441990  0.085584
97  15.893987  0.072568
98  18.282501  2.253100
99  15.590181  0.041649

[100 rows x 2 columns]
```
```
<class 'pandas.core.series.Series'> objid
<class 'pandas.core.series.Series'> ra
<class 'pandas.core.series.Series'> dec
<class 'pandas.core.series.Series'> u
<class 'pandas.core.series.Series'> g
<class 'pandas.core.series.Series'> r
<class 'pandas.core.series.Series'> i
<class 'pandas.core.frame.DataFrame'> z
<class 'pandas.core.series.Series'> run
<class 'pandas.core.series.Series'> rerun
<class 'pandas.core.series.Series'> camcol
<class 'pandas.core.series.Series'> field
<class 'pandas.core.series.Series'> specobjid
<class 'pandas.core.series.Series'> specclass
<class 'pandas.core.frame.DataFrame'> z
<class 'pandas.core.series.Series'> plate
<class 'pandas.core.series.Series'> mjd
```


## don't edit your python files when they are *copied* into your modules when debugging

this causes breakpoints not to run and your stacktraces to be plain wrong