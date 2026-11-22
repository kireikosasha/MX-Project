"""Bluetooth/bluetooth_reduced.dot: a Mealy machine learned by others from a real implementation (AALpy, DotModels, MIT licence).
3 states, 9 inputs. The answer to a story of inputs is the output to the last of them."""

ALPHABET = ['connection_req', 'feature_req', 'feature_rsp', 'length_req', 'length_rsp', 'mtu_req', 'pairing_req', 'scan_req', 'version_req']
START = 's0'
TRANS = {
    ('s0', 'connection_req'): ('s1', 'BTLE|BTLE_DATA'),
    ('s0', 'feature_req'): ('s0', 'Empty'),
    ('s0', 'feature_rsp'): ('s0', 'Empty'),
    ('s0', 'length_req'): ('s0', 'Empty'),
    ('s0', 'length_rsp'): ('s0', 'Empty'),
    ('s0', 'mtu_req'): ('s0', 'Empty'),
    ('s0', 'pairing_req'): ('s0', 'Empty'),
    ('s0', 'scan_req'): ('s0', 'Adv'),
    ('s0', 'version_req'): ('s0', 'Empty'),
    ('s1', 'connection_req'): ('s1', 'BTLE|BTLE_DATA'),
    ('s1', 'feature_req'): ('s1', 'BTLE|BTLE_CTRL|BTLE_DATA|LL_FEATURE_RSP'),
    ('s1', 'feature_rsp'): ('s1', 'BTLE|BTLE_CTRL|BTLE_DATA|LL_REJECT_IND'),
    ('s1', 'length_req'): ('s1', 'BTLE|BTLE_CTRL|BTLE_DATA|LL_UNKNOWN_RSP'),
    ('s1', 'length_rsp'): ('s1', 'BTLE|BTLE_CTRL|BTLE_DATA|LL_UNKNOWN_RSP'),
    ('s1', 'mtu_req'): ('s1', 'ATT_Exchange_MTU_Response|ATT_Hdr|BTLE|BTLE_DATA|L2CAP_Hdr'),
    ('s1', 'pairing_req'): ('s1', 'BTLE|BTLE_DATA|L2CAP_Hdr|SM_Failed|SM_Hdr'),
    ('s1', 'scan_req'): ('s0', 'Adv'),
    ('s1', 'version_req'): ('s2', 'BTLE|BTLE_CTRL|BTLE_DATA|LL_VERSION_IND'),
    ('s2', 'connection_req'): ('s1', 'BTLE|BTLE_DATA'),
    ('s2', 'feature_req'): ('s2', 'BTLE|BTLE_CTRL|BTLE_DATA|LL_FEATURE_RSP'),
    ('s2', 'feature_rsp'): ('s2', 'BTLE|BTLE_CTRL|BTLE_DATA|LL_REJECT_IND'),
    ('s2', 'length_req'): ('s2', 'BTLE|BTLE_CTRL|BTLE_DATA|LL_UNKNOWN_RSP'),
    ('s2', 'length_rsp'): ('s2', 'BTLE|BTLE_CTRL|BTLE_DATA|LL_UNKNOWN_RSP'),
    ('s2', 'mtu_req'): ('s2', 'ATT_Exchange_MTU_Response|ATT_Hdr|BTLE|BTLE_DATA|L2CAP_Hdr'),
    ('s2', 'pairing_req'): ('s2', 'BTLE|BTLE_DATA|L2CAP_Hdr|SM_Failed|SM_Hdr'),
    ('s2', 'scan_req'): ('s0', 'Adv'),
    ('s2', 'version_req'): ('s2', 'BTLE|BTLE_DATA'),
}


def answer(seq):
    q, out = START, ""
    for a in seq:
        q, out = TRANS[q, a]
    return out
