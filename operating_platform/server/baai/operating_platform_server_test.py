import operating_platform_server

if __name__ == '__main__':
    server = operating_platform_server.FlaskServer()
    server.run()



# python -m nuitka --module core_module.py