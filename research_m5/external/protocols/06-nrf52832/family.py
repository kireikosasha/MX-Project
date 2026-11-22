"""Bluetooth/nRF52832.dot: a Mealy machine learned by others from a real implementation (AALpy, DotModels, MIT licence).
5 states, 9 inputs. The answer to a story of inputs is the output to the last of them."""

ALPHABET = ['connection_req', 'feature_req', 'feature_rsp', 'length_req', 'length_rsp', 'mtu_req', 'pairing_req', 'scan_req', 'version_req']
START = 's0'
TRANS = {
    ('s0', 'connection_req'): ('s1', 'BTLE|BTLE_DATA|L2CAP_Hdr|Raw|SM_Hdr'),
    ('s0', 'feature_req'): ('s0', 'Empty'),
    ('s0', 'feature_rsp'): ('s0', 'Empty'),
    ('s0', 'length_req'): ('s0', 'Empty'),
    ('s0', 'length_rsp'): ('s0', 'Empty'),
    ('s0', 'mtu_req'): ('s0', 'Empty'),
    ('s0', 'pairing_req'): ('s0', 'Empty'),
    ('s0', 'scan_req'): ('s0', 'Adv'),
    ('s0', 'version_req'): ('s0', 'Empty'),
    ('s1', 'connection_req'): ('s1', 'BTLE|BTLE_DATA|L2CAP_Hdr|Raw|SM_Hdr'),
    ('s1', 'feature_req'): ('s1', 'BTLE|BTLE_CTRL|BTLE_DATA|LL_FEATURE_RSP'),
    ('s1', 'feature_rsp'): ('s1', 'BTLE|BTLE_CTRL|BTLE_DATA|LL_UNKNOWN_RSP'),
    ('s1', 'length_req'): ('s1', 'BTLE|BTLE_CTRL|BTLE_DATA|LL_LENGTH_RSP'),
    ('s1', 'length_rsp'): ('s0', 'BTLE|BTLE_DATA'),
    ('s1', 'mtu_req'): ('s2', 'ATT_Exchange_MTU_Response|ATT_Hdr|BTLE|BTLE_DATA|L2CAP_Hdr'),
    ('s1', 'pairing_req'): ('s1', 'BTLE|BTLE_DATA|L2CAP_Hdr|SM_Hdr|SM_Pairing_Response'),
    ('s1', 'scan_req'): ('s0', 'Adv'),
    ('s1', 'version_req'): ('s3', 'BTLE|BTLE_CTRL|BTLE_DATA|LL_VERSION_IND'),
    ('s2', 'connection_req'): ('s1', 'BTLE|BTLE_DATA|L2CAP_Hdr|Raw|SM_Hdr'),
    ('s2', 'feature_req'): ('s2', 'BTLE|BTLE_CTRL|BTLE_DATA|LL_FEATURE_RSP'),
    ('s2', 'feature_rsp'): ('s2', 'BTLE|BTLE_CTRL|BTLE_DATA|LL_UNKNOWN_RSP'),
    ('s2', 'length_req'): ('s2', 'BTLE|BTLE_CTRL|BTLE_DATA|LL_LENGTH_RSP'),
    ('s2', 'length_rsp'): ('s0', 'BTLE|BTLE_DATA'),
    ('s2', 'mtu_req'): ('s2', 'ATT_Error_Response|ATT_Hdr|BTLE|BTLE_DATA|L2CAP_Hdr'),
    ('s2', 'pairing_req'): ('s2', 'BTLE|BTLE_DATA|L2CAP_Hdr|SM_Hdr|SM_Pairing_Response'),
    ('s2', 'scan_req'): ('s0', 'Adv'),
    ('s2', 'version_req'): ('s4', 'BTLE|BTLE_CTRL|BTLE_DATA|LL_VERSION_IND'),
    ('s3', 'connection_req'): ('s1', 'BTLE|BTLE_DATA|L2CAP_Hdr|Raw|SM_Hdr'),
    ('s3', 'feature_req'): ('s3', 'BTLE|BTLE_CTRL|BTLE_DATA|LL_FEATURE_RSP'),
    ('s3', 'feature_rsp'): ('s3', 'BTLE|BTLE_CTRL|BTLE_DATA|LL_UNKNOWN_RSP'),
    ('s3', 'length_req'): ('s3', 'BTLE|BTLE_CTRL|BTLE_DATA|LL_LENGTH_RSP'),
    ('s3', 'length_rsp'): ('s0', 'BTLE|BTLE_DATA'),
    ('s3', 'mtu_req'): ('s4', 'ATT_Exchange_MTU_Response|ATT_Hdr|BTLE|BTLE_DATA|L2CAP_Hdr'),
    ('s3', 'pairing_req'): ('s3', 'BTLE|BTLE_DATA|L2CAP_Hdr|SM_Hdr|SM_Pairing_Response'),
    ('s3', 'scan_req'): ('s0', 'Adv'),
    ('s3', 'version_req'): ('s3', 'BTLE|BTLE_DATA'),
    ('s4', 'connection_req'): ('s1', 'BTLE|BTLE_DATA|L2CAP_Hdr|Raw|SM_Hdr'),
    ('s4', 'feature_req'): ('s4', 'BTLE|BTLE_CTRL|BTLE_DATA|LL_FEATURE_RSP'),
    ('s4', 'feature_rsp'): ('s4', 'BTLE|BTLE_CTRL|BTLE_DATA|LL_UNKNOWN_RSP'),
    ('s4', 'length_req'): ('s4', 'BTLE|BTLE_CTRL|BTLE_DATA|LL_LENGTH_RSP'),
    ('s4', 'length_rsp'): ('s0', 'BTLE|BTLE_DATA'),
    ('s4', 'mtu_req'): ('s4', 'ATT_Error_Response|ATT_Hdr|BTLE|BTLE_DATA|L2CAP_Hdr'),
    ('s4', 'pairing_req'): ('s4', 'BTLE|BTLE_DATA|L2CAP_Hdr|SM_Hdr|SM_Pairing_Response'),
    ('s4', 'scan_req'): ('s0', 'Adv'),
    ('s4', 'version_req'): ('s4', 'BTLE|BTLE_DATA'),
}


def answer(seq):
    q, out = START, ""
    for a in seq:
        q, out = TRANS[q, a]
    return out
