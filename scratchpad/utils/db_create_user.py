import sys

import oracledb

# Update with your username, password, hostname, and service_name
username = "system"
password = "Welcome1"
dsn = "ramgudla.ad2.devintegratiphx.oraclevcn.com:1521/FREEPDB1"

try:
    conn = oracledb.connect(user=username, password=password, dsn=dsn)
    print("Connection successful!")

    cursor = conn.cursor()
    try:
        cursor.execute(
            """
            begin
                -- Drop user
                begin
                    execute immediate 'drop user testuser cascade';
                exception
                    when others then
                        dbms_output.put_line('Error dropping user: ' || SQLERRM);
                end;

                -- Create user and grant privileges
                execute immediate 'create user testuser identified by testuser';
                execute immediate 'grant connect, unlimited tablespace, create credential, create procedure, create any index to testuser';
                execute immediate 'create or replace directory DEMO_PY_DIR as ''/scratch/hroy/view_storage/hroy_devstorage/demo/orachain''';
                execute immediate 'grant read, write on directory DEMO_PY_DIR to public';
                execute immediate 'grant create mining model to testuser';

                -- Network access
                begin
                    DBMS_NETWORK_ACL_ADMIN.APPEND_HOST_ACE(
                        host => '*',
                        ace => xs$ace_type(privilege_list => xs$name_list('connect'),
                                           principal_name => 'testuser',
                                           principal_type => xs_acl.ptype_db)
                    );
                end;
            end;
            """
        )
        print("User setup done!")
    except Exception as e:
        print(f"User setup failed with error: {e}")
    finally:
        cursor.close()
    conn.close()
except Exception as e:
    print(f"Connection failed with error: {e}")
    sys.exit(1)
