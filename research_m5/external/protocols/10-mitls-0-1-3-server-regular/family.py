"""TLS/miTLS_0.1.3_server_regular.dot: a Mealy machine learned by others from a real implementation (AALpy, DotModels, MIT licence).
6 states, 8 inputs. The answer to a story of inputs is the output to the last of them."""

ALPHABET = ['ApplicationData', 'ApplicationDataEmpty', 'ChangeCipherSpec', 'ClientHelloRSA', 'ClientKeyExchange', 'EmptyCertificate', 'Finished', 'HeartbeatRequest']
START = '2'
TRANS = {
    ('0', 'ApplicationData'): ('5', 'ApplicationDataApplicationDataApplicationDataApplicationDataApplicationDataApplicationData & ConnectionClosed'),
    ('0', 'ApplicationDataEmpty'): ('0', 'Empty'),
    ('0', 'ChangeCipherSpec'): ('5', 'Alert Fatal (Unexpected message) & ConnectionClosed'),
    ('0', 'ClientHelloRSA'): ('5', 'Alert Fatal (Handshake failure) & ConnectionClosed'),
    ('0', 'ClientKeyExchange'): ('5', 'Alert Fatal (Unexpected message) & ConnectionClosed'),
    ('0', 'EmptyCertificate'): ('5', 'Alert Fatal (Unexpected message) & ConnectionClosed'),
    ('0', 'Finished'): ('5', 'Alert Fatal (Unexpected message) & ConnectionClosed'),
    ('0', 'HeartbeatRequest'): ('5', 'ConnectionClosed'),
    ('1', 'ApplicationData'): ('5', 'ConnectionClosed'),
    ('1', 'ApplicationDataEmpty'): ('5', 'Alert Fatal (Illegal parameter) & ConnectionClosed'),
    ('1', 'ChangeCipherSpec'): ('5', 'Alert Fatal (Unexpected message) & ConnectionClosed'),
    ('1', 'ClientHelloRSA'): ('5', 'Alert Fatal (Unexpected message) & ConnectionClosed'),
    ('1', 'ClientKeyExchange'): ('3', 'Empty'),
    ('1', 'EmptyCertificate'): ('5', 'Alert Fatal (Unexpected message) & ConnectionClosed'),
    ('1', 'Finished'): ('5', 'Alert Fatal (Unexpected message) & ConnectionClosed'),
    ('1', 'HeartbeatRequest'): ('5', 'ConnectionClosed'),
    ('2', 'ApplicationData'): ('5', 'ConnectionClosed'),
    ('2', 'ApplicationDataEmpty'): ('5', 'Alert Fatal (Illegal parameter) & ConnectionClosed'),
    ('2', 'ChangeCipherSpec'): ('5', 'ConnectionClosed'),
    ('2', 'ClientHelloRSA'): ('1', 'ServerHello & Certificate & ServerHelloDone'),
    ('2', 'ClientKeyExchange'): ('5', 'Alert Fatal (Unexpected message) & ConnectionClosed'),
    ('2', 'EmptyCertificate'): ('5', 'Alert Fatal (Unexpected message) & ConnectionClosed'),
    ('2', 'Finished'): ('5', 'Alert Fatal (Unexpected message) & ConnectionClosed'),
    ('2', 'HeartbeatRequest'): ('5', 'ConnectionClosed'),
    ('3', 'ApplicationData'): ('5', 'ConnectionClosed'),
    ('3', 'ApplicationDataEmpty'): ('5', 'Alert Fatal (Illegal parameter) & ConnectionClosed'),
    ('3', 'ChangeCipherSpec'): ('4', 'Empty'),
    ('3', 'ClientHelloRSA'): ('5', 'Alert Fatal (Unexpected message) & ConnectionClosed'),
    ('3', 'ClientKeyExchange'): ('5', 'Alert Fatal (Unexpected message) & ConnectionClosed'),
    ('3', 'EmptyCertificate'): ('5', 'Alert Fatal (Unexpected message) & ConnectionClosed'),
    ('3', 'Finished'): ('5', 'Alert Fatal (Unexpected message) & ConnectionClosed'),
    ('3', 'HeartbeatRequest'): ('5', 'ConnectionClosed'),
    ('4', 'ApplicationData'): ('5', 'ConnectionClosed'),
    ('4', 'ApplicationDataEmpty'): ('5', 'ConnectionClosed'),
    ('4', 'ChangeCipherSpec'): ('5', 'ConnectionClosed'),
    ('4', 'ClientHelloRSA'): ('5', 'Alert Fatal (Unexpected message) & ConnectionClosed'),
    ('4', 'ClientKeyExchange'): ('5', 'Alert Fatal (Unexpected message) & ConnectionClosed'),
    ('4', 'EmptyCertificate'): ('5', 'Alert Fatal (Unexpected message) & ConnectionClosed'),
    ('4', 'Finished'): ('0', 'ChangeCipherSpec & Finished'),
    ('4', 'HeartbeatRequest'): ('5', 'ConnectionClosed'),
    ('5', 'ApplicationData'): ('5', 'ConnectionClosed'),
    ('5', 'ApplicationDataEmpty'): ('5', 'ConnectionClosed'),
    ('5', 'ChangeCipherSpec'): ('5', 'ConnectionClosed'),
    ('5', 'ClientHelloRSA'): ('5', 'ConnectionClosed'),
    ('5', 'ClientKeyExchange'): ('5', 'ConnectionClosed'),
    ('5', 'EmptyCertificate'): ('5', 'ConnectionClosed'),
    ('5', 'Finished'): ('5', 'ConnectionClosed'),
    ('5', 'HeartbeatRequest'): ('5', 'ConnectionClosed'),
}


def answer(seq):
    q, out = START, ""
    for a in seq:
        q, out = TRANS[q, a]
    return out
