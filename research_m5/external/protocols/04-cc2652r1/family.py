"""Bluetooth/cc2652r1.dot: a Mealy machine learned by others from a real implementation (AALpy, DotModels, MIT licence).
4 states, 7 inputs. The answer to a story of inputs is the output to the last of them."""

ALPHABET = ['feature_req', 'feature_rsp', 'length_req', 'length_rsp', 'mtu_req', 'pairing_req', 'version_req']
START = 's0'
TRANS = {
    ('s0', 'feature_req'): ('s0', 'BTLE|BTLE_CTRL|BTLE_DATA|LL_FEATURE_RSP'),
    ('s0', 'feature_rsp'): ('s2', 'BTLE|BTLE_CTRL|BTLE_DATA|LL_LENGTH_REQ'),
    ('s0', 'length_req'): ('s0', 'BTLE|BTLE_CTRL|BTLE_DATA|LL_LENGTH_RSP'),
    ('s0', 'length_rsp'): ('s0', 'BTLE|BTLE_DATA'),
    ('s0', 'mtu_req'): ('s0', 'ATT_Exchange_MTU_Response|ATT_Hdr|BTLE|BTLE_DATA|L2CAP_Hdr'),
    ('s0', 'pairing_req'): ('s1', 'BTLE|BTLE_DATA|L2CAP_Hdr|SM_Hdr|SM_Pairing_Response'),
    ('s0', 'version_req'): ('s0', 'BTLE|BTLE_CTRL|BTLE_DATA|LL_VERSION_IND'),
    ('s1', 'feature_req'): ('s1', 'BTLE|BTLE_CTRL|BTLE_DATA|LL_FEATURE_RSP'),
    ('s1', 'feature_rsp'): ('s3', 'BTLE|BTLE_CTRL|BTLE_DATA|LL_LENGTH_REQ'),
    ('s1', 'length_req'): ('s1', 'BTLE|BTLE_CTRL|BTLE_DATA|LL_LENGTH_RSP'),
    ('s1', 'length_rsp'): ('s1', 'BTLE|BTLE_DATA'),
    ('s1', 'mtu_req'): ('s1', 'ATT_Exchange_MTU_Response|ATT_Hdr|BTLE|BTLE_DATA|L2CAP_Hdr'),
    ('s1', 'pairing_req'): ('s0', 'BTLE|BTLE_DATA|L2CAP_Hdr|SM_Failed|SM_Hdr'),
    ('s1', 'version_req'): ('s1', 'BTLE|BTLE_CTRL|BTLE_DATA|LL_VERSION_IND'),
    ('s2', 'feature_req'): ('s2', 'BTLE|BTLE_CTRL|BTLE_DATA|LL_FEATURE_RSP'),
    ('s2', 'feature_rsp'): ('s2', 'BTLE|BTLE_DATA'),
    ('s2', 'length_req'): ('s2', 'BTLE|BTLE_CTRL|BTLE_DATA|LL_LENGTH_RSP'),
    ('s2', 'length_rsp'): ('s0', 'BTLE|BTLE_DATA'),
    ('s2', 'mtu_req'): ('s2', 'ATT_Exchange_MTU_Response|ATT_Hdr|BTLE|BTLE_DATA|L2CAP_Hdr'),
    ('s2', 'pairing_req'): ('s3', 'BTLE|BTLE_DATA|L2CAP_Hdr|SM_Hdr|SM_Pairing_Response'),
    ('s2', 'version_req'): ('s2', 'BTLE|BTLE_CTRL|BTLE_DATA|LL_VERSION_IND'),
    ('s3', 'feature_req'): ('s3', 'BTLE|BTLE_CTRL|BTLE_DATA|LL_FEATURE_RSP'),
    ('s3', 'feature_rsp'): ('s3', 'BTLE|BTLE_DATA'),
    ('s3', 'length_req'): ('s3', 'BTLE|BTLE_CTRL|BTLE_DATA|LL_LENGTH_RSP'),
    ('s3', 'length_rsp'): ('s1', 'BTLE|BTLE_DATA'),
    ('s3', 'mtu_req'): ('s3', 'ATT_Exchange_MTU_Response|ATT_Hdr|BTLE|BTLE_DATA|L2CAP_Hdr'),
    ('s3', 'pairing_req'): ('s2', 'BTLE|BTLE_DATA|L2CAP_Hdr|SM_Failed|SM_Hdr'),
    ('s3', 'version_req'): ('s3', 'BTLE|BTLE_CTRL|BTLE_DATA|LL_VERSION_IND'),
}


def answer(seq):
    q, out = START, ""
    for a in seq:
        q, out = TRANS[q, a]
    return out
