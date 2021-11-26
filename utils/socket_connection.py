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


def recieve_data():

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect(('127.0.0.1', 25001))

    while True:
        data = s.recv(1024)
        print(data.decode())
    # data = s.recv(1024)
    # print(data.decode())


if __name__ == '__main__':
    recieve_data()