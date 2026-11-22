"""Bluetooth/CC2640R2-no-pairing-req.dot: a Mealy machine learned by others from a real implementation (AALpy, DotModels, MIT licence).
6 states, 8 inputs. The answer to a story of inputs is the output to the last of them."""

ALPHABET = ['connection_req', 'feature_req', 'feature_rsp', 'length_req', 'length_rsp', 'mtu_req', 'scan_req', 'version_req']
START = 's0'
TRANS = {
    ('s0', 'connection_req'): ('s1', 'BTLE|BTLE_CTRL|BTLE_DATA|LL_LENGTH_REQ'),
    ('s0', 'feature_req'): ('s0', 'Empty'),
    ('s0', 'feature_rsp'): ('s0', 'Empty'),
    ('s0', 'length_req'): ('s0', 'Empty'),
    ('s0', 'length_rsp'): ('s0', 'Empty'),
    ('s0', 'mtu_req'): ('s0', 'Empty'),
    ('s0', 'scan_req'): ('s0', 'Adv'),
    ('s0', 'version_req'): ('s0', 'Empty'),
    ('s1', 'connection_req'): ('s1', 'BTLE|BTLE_CTRL|BTLE_DATA|LL_LENGTH_REQ'),
    ('s1', 'feature_req'): ('s1', 'BTLE|BTLE_CTRL|BTLE_DATA|LL_FEATURE_RSP'),
    ('s1', 'feature_rsp'): ('s1', 'BTLE|BTLE_DATA'),
    ('s1', 'length_req'): ('s1', 'BTLE|BTLE_CTRL|BTLE_DATA|LL_LENGTH_RSP'),
    ('s1', 'length_rsp'): ('s3', 'BTLE|BTLE_DATA'),
    ('s1', 'mtu_req'): ('s1', 'ATT_Exchange_MTU_Response|ATT_Hdr|BTLE|BTLE_DATA|L2CAP_Hdr'),
    ('s1', 'scan_req'): ('s0', 'Adv'),
    ('s1', 'version_req'): ('s2', 'BTLE|BTLE_DATA'),
    ('s2', 'connection_req'): ('s1', 'BTLE|BTLE_CTRL|BTLE_DATA|LL_LENGTH_REQ'),
    ('s2', 'feature_req'): ('s2', 'BTLE|BTLE_CTRL|BTLE_DATA|LL_FEATURE_RSP'),
    ('s2', 'feature_rsp'): ('s2', 'BTLE|BTLE_DATA'),
    ('s2', 'length_req'): ('s2', 'BTLE|BTLE_CTRL|BTLE_DATA|LL_LENGTH_RSP'),
    ('s2', 'length_rsp'): ('s5', 'BTLE|BTLE_CTRL|BTLE_DATA|LL_VERSION_IND'),
    ('s2', 'mtu_req'): ('s2', 'ATT_Exchange_MTU_Response|ATT_Hdr|BTLE|BTLE_DATA|L2CAP_Hdr'),
    ('s2', 'scan_req'): ('s0', 'Adv'),
    ('s2', 'version_req'): ('s4', 'BTLE|BTLE_DATA'),
    ('s3', 'connection_req'): ('s1', 'BTLE|BTLE_CTRL|BTLE_DATA|LL_LENGTH_REQ'),
    ('s3', 'feature_req'): ('s3', 'BTLE|BTLE_CTRL|BTLE_DATA|LL_FEATURE_RSP'),
    ('s3', 'feature_rsp'): ('s3', 'BTLE|BTLE_DATA'),
    ('s3', 'length_req'): ('s3', 'BTLE|BTLE_CTRL|BTLE_DATA|LL_LENGTH_RSP'),
    ('s3', 'length_rsp'): ('s3', 'BTLE|BTLE_DATA'),
    ('s3', 'mtu_req'): ('s3', 'ATT_Exchange_MTU_Response|ATT_Hdr|BTLE|BTLE_DATA|L2CAP_Hdr'),
    ('s3', 'scan_req'): ('s0', 'Adv'),
    ('s3', 'version_req'): ('s5', 'BTLE|BTLE_CTRL|BTLE_DATA|LL_VERSION_IND'),
    ('s4', 'connection_req'): ('s1', 'BTLE|BTLE_CTRL|BTLE_DATA|LL_LENGTH_REQ'),
    ('s4', 'feature_req'): ('s4', 'BTLE|BTLE_CTRL|BTLE_DATA|LL_FEATURE_RSP'),
    ('s4', 'feature_rsp'): ('s4', 'BTLE|BTLE_DATA'),
    ('s4', 'length_req'): ('s4', 'BTLE|BTLE_CTRL|BTLE_DATA|LL_LENGTH_RSP'),
    ('s4', 'length_rsp'): ('s5', 'BTLE|BTLE_CTRL|BTLE_DATA|LL_UNKNOWN_RSP|LL_VERSION_IND'),
    ('s4', 'mtu_req'): ('s4', 'ATT_Exchange_MTU_Response|ATT_Hdr|BTLE|BTLE_DATA|L2CAP_Hdr'),
    ('s4', 'scan_req'): ('s0', 'Adv'),
    ('s4', 'version_req'): ('s4', 'BTLE|BTLE_DATA'),
    ('s5', 'connection_req'): ('s1', 'BTLE|BTLE_CTRL|BTLE_DATA|LL_LENGTH_REQ'),
    ('s5', 'feature_req'): ('s5', 'BTLE|BTLE_CTRL|BTLE_DATA|LL_FEATURE_RSP'),
    ('s5', 'feature_rsp'): ('s5', 'BTLE|BTLE_DATA'),
    ('s5', 'length_req'): ('s5', 'BTLE|BTLE_CTRL|BTLE_DATA|LL_LENGTH_RSP'),
    ('s5', 'length_rsp'): ('s5', 'BTLE|BTLE_DATA'),
    ('s5', 'mtu_req'): ('s5', 'ATT_Exchange_MTU_Response|ATT_Hdr|BTLE|BTLE_DATA|L2CAP_Hdr'),
    ('s5', 'scan_req'): ('s0', 'Adv'),
    ('s5', 'version_req'): ('s5', 'BTLE|BTLE_CTRL|BTLE_DATA|LL_UNKNOWN_RSP'),
}


def answer(seq):
    q, out = START, ""
    for a in seq:
        q, out = TRANS[q, a]
    return out
