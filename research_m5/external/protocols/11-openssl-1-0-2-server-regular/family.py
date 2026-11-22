"""TLS/OpenSSL_1.0.2_server_regular.dot: a Mealy machine learned by others from a real implementation (AALpy, DotModels, MIT licence).
7 states, 7 inputs. The answer to a story of inputs is the output to the last of them."""

ALPHABET = ['ApplicationData', 'ApplicationDataEmpty', 'ChangeCipherSpec', 'ClientHelloRSA', 'ClientKeyExchange', 'EmptyCertificate', 'Finished']
START = '6'
TRANS = {
    ('0', 'ApplicationData'): ('4', 'Alert Fatal (Unexpected message) & ConnectionClosed'),
    ('0', 'ApplicationDataEmpty'): ('0', 'Empty'),
    ('0', 'ChangeCipherSpec'): ('4', 'Alert Fatal (Unexpected message) & ConnectionClosed'),
    ('0', 'ClientHelloRSA'): ('4', 'Alert Fatal (Unexpected message) & ConnectionClosed'),
    ('0', 'ClientKeyExchange'): ('4', 'Alert Fatal (Unexpected message) & ConnectionClosed'),
    ('0', 'EmptyCertificate'): ('4', 'Alert Fatal (Unexpected message) & ConnectionClosed'),
    ('0', 'Finished'): ('3', 'ChangeCipherSpec & Finished'),
    ('1', 'ApplicationData'): ('4', 'Alert Fatal (Unexpected message) & ConnectionClosed'),
    ('1', 'ApplicationDataEmpty'): ('1', 'Empty'),
    ('1', 'ChangeCipherSpec'): ('4', 'Alert Fatal (Unexpected message) & ConnectionClosed'),
    ('1', 'ClientHelloRSA'): ('4', 'Alert Fatal (Unexpected message) & ConnectionClosed'),
    ('1', 'ClientKeyExchange'): ('2', 'Empty'),
    ('1', 'EmptyCertificate'): ('4', 'Alert Fatal (Unexpected message) & ConnectionClosed'),
    ('1', 'Finished'): ('4', 'Alert Fatal (Unexpected message) & ConnectionClosed'),
    ('2', 'ApplicationData'): ('4', 'Alert Fatal (Unexpected message) & ConnectionClosed'),
    ('2', 'ApplicationDataEmpty'): ('2', 'Empty'),
    ('2', 'ChangeCipherSpec'): ('0', 'Empty'),
    ('2', 'ClientHelloRSA'): ('4', 'Alert Fatal (Unexpected message) & ConnectionClosed'),
    ('2', 'ClientKeyExchange'): ('4', 'Alert Fatal (Unexpected message) & ConnectionClosed'),
    ('2', 'EmptyCertificate'): ('4', 'Alert Fatal (Unexpected message) & ConnectionClosed'),
    ('2', 'Finished'): ('4', 'Alert Fatal (Unexpected message) & ConnectionClosed'),
    ('3', 'ApplicationData'): ('4', 'ApplicationData & ConnectionClosed'),
    ('3', 'ApplicationDataEmpty'): ('3', 'Empty'),
    ('3', 'ChangeCipherSpec'): ('4', 'Alert Fatal (Unexpected message) & ConnectionClosed'),
    ('3', 'ClientHelloRSA'): ('4', 'Alert Fatal (Handshake failure) & ConnectionClosed'),
    ('3', 'ClientKeyExchange'): ('4', 'Alert Fatal (Unexpected message) & ConnectionClosed'),
    ('3', 'EmptyCertificate'): ('4', 'Alert Fatal (Unexpected message) & ConnectionClosed'),
    ('3', 'Finished'): ('4', 'Alert Fatal (Unexpected message) & ConnectionClosed'),
    ('4', 'ApplicationData'): ('4', 'ConnectionClosed'),
    ('4', 'ApplicationDataEmpty'): ('4', 'ConnectionClosed'),
    ('4', 'ChangeCipherSpec'): ('4', 'ConnectionClosed'),
    ('4', 'ClientHelloRSA'): ('4', 'ConnectionClosed'),
    ('4', 'ClientKeyExchange'): ('4', 'ConnectionClosed'),
    ('4', 'EmptyCertificate'): ('4', 'ConnectionClosed'),
    ('4', 'Finished'): ('4', 'ConnectionClosed'),
    ('5', 'ApplicationData'): ('4', 'ConnectionClosed'),
    ('5', 'ApplicationDataEmpty'): ('4', 'Empty'),
    ('5', 'ChangeCipherSpec'): ('4', 'ConnectionClosed'),
    ('5', 'ClientHelloRSA'): ('4', 'ConnectionClosed'),
    ('5', 'ClientKeyExchange'): ('4', 'ConnectionClosed'),
    ('5', 'EmptyCertificate'): ('4', 'ConnectionClosed'),
    ('5', 'Finished'): ('4', 'ConnectionClosed'),
    ('6', 'ApplicationData'): ('4', 'ConnectionClosed'),
    ('6', 'ApplicationDataEmpty'): ('5', 'Empty'),
    ('6', 'ChangeCipherSpec'): ('4', 'Empty'),
    ('6', 'ClientHelloRSA'): ('1', 'ServerHello & Certificate & ServerHelloDone'),
    ('6', 'ClientKeyExchange'): ('4', 'ConnectionClosed'),
    ('6', 'EmptyCertificate'): ('4', 'ConnectionClosed'),
    ('6', 'Finished'): ('4', 'ConnectionClosed'),
}


def answer(seq):
    q, out = START, ""
    for a in seq:
        q, out = TRANS[q, a]
    return out
