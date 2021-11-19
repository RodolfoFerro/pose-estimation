import socket


def create_socket_connection(host, port):

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind((host, port))
    s.listen(5)


    # Start connection
    print('[INFO] Starting conection')
    conn, addr = s.accept()
    print('[INFO] Connection established with: ', addr)

    return conn