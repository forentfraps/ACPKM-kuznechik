import numpy as np
from functools import reduce
from collections import deque
import os
import binascii

SBOX = [
    0xFC, 0xEE, 0xDD, 0x11, 0xCF, 0x6E, 0x31, 0x16, 0xFB, 0xC4, 0xFA, 0xDA,
    0x23, 0xC5, 0x04, 0x4D, 0xE9, 0x77, 0xF0, 0xDB, 0x93, 0x2E, 0x99, 0xBA,
    0x17, 0x36, 0xF1, 0xBB, 0x14, 0xCD, 0x5F, 0xC1, 0xF9, 0x18, 0x65, 0x5A,
    0xE2, 0x5C, 0xEF, 0x21, 0x81, 0x1C, 0x3C, 0x42, 0x8B, 0x01, 0x8E, 0x4F,
    0x05, 0x84, 0x02, 0xAE, 0xE3, 0x6A, 0x8F, 0xA0, 0x06, 0x0B, 0xED, 0x98,
    0x7F, 0xD4, 0xD3, 0x1F, 0xEB, 0x34, 0x2C, 0x51, 0xEA, 0xC8, 0x48, 0xAB,
    0xF2, 0x2A, 0x68, 0xA2, 0xFD, 0x3A, 0xCE, 0xCC, 0xB5, 0x70, 0x0E, 0x56,
    0x08, 0x0C, 0x76, 0x12, 0xBF, 0x72, 0x13, 0x47, 0x9C, 0xB7, 0x5D, 0x87,
    0x15, 0xA1, 0x96, 0x29, 0x10, 0x7B, 0x9A, 0xC7, 0xF3, 0x91, 0x78, 0x6F,
    0x9D, 0x9E, 0xB2, 0xB1, 0x32, 0x75, 0x19, 0x3D, 0xFF, 0x35, 0x8A, 0x7E,
    0x6D, 0x54, 0xC6, 0x80, 0xC3, 0xBD, 0x0D, 0x57, 0xDF, 0xF5, 0x24, 0xA9,
    0x3E, 0xA8, 0x43, 0xC9, 0xD7, 0x79, 0xD6, 0xF6, 0x7C, 0x22, 0xB9, 0x03,
    0xE0, 0x0F, 0xEC, 0xDE, 0x7A, 0x94, 0xB0, 0xBC, 0xDC, 0xE8, 0x28, 0x50,
    0x4E, 0x33, 0x0A, 0x4A, 0xA7, 0x97, 0x60, 0x73, 0x1E, 0x00, 0x62, 0x44,
    0x1A, 0xB8, 0x38, 0x82, 0x64, 0x9F, 0x26, 0x41, 0xAD, 0x45, 0x46, 0x92,
    0x27, 0x5E, 0x55, 0x2F, 0x8C, 0xA3, 0xA5, 0x7D, 0x69, 0xD5, 0x95, 0x3B,
    0x07, 0x58, 0xB3, 0x40, 0x86, 0xAC, 0x1D, 0xF7, 0x30, 0x37, 0x6B, 0xE4,
    0x88, 0xD9, 0xE7, 0x89, 0xE1, 0x1B, 0x83, 0x49, 0x4C, 0x3F, 0xF8, 0xFE,
    0x8D, 0x53, 0xAA, 0x90, 0xCA, 0xD8, 0x85, 0x61, 0x20, 0x71, 0x67, 0xA4,
    0x2D, 0x2B, 0x09, 0x5B, 0xCB, 0x9B, 0x25, 0xD0, 0xBE, 0xE5, 0x6C, 0x52,
    0x59, 0xA6, 0x74, 0xD2, 0xE6, 0xF4, 0xB4, 0xC0, 0xD1, 0x66, 0xAF, 0xC2,
    0x39, 0x4B, 0x63, 0xB6
]

RESBOX = [
    0xA5, 0x2D, 0x32, 0x8F, 0x0E, 0x30, 0x38, 0xC0, 0x54, 0xE6, 0x9E, 0x39,
    0x55, 0x7E, 0x52, 0x91, 0x64, 0x03, 0x57, 0x5A, 0x1C, 0x60, 0x07, 0x18,
    0x21, 0x72, 0xA8, 0xD1, 0x29, 0xC6, 0xA4, 0x3F, 0xE0, 0x27, 0x8D, 0x0C,
    0x82, 0xEA, 0xAE, 0xB4, 0x9A, 0x63, 0x49, 0xE5, 0x42, 0xE4, 0x15, 0xB7,
    0xC8, 0x06, 0x70, 0x9D, 0x41, 0x75, 0x19, 0xC9, 0xAA, 0xFC, 0x4D, 0xBF,
    0x2A, 0x73, 0x84, 0xD5, 0xC3, 0xAF, 0x2B, 0x86, 0xA7, 0xB1, 0xB2, 0x5B,
    0x46, 0xD3, 0x9F, 0xFD, 0xD4, 0x0F, 0x9C, 0x2F, 0x9B, 0x43, 0xEF, 0xD9,
    0x79, 0xB6, 0x53, 0x7F, 0xC1, 0xF0, 0x23, 0xE7, 0x25, 0x5E, 0xB5, 0x1E,
    0xA2, 0xDF, 0xA6, 0xFE, 0xAC, 0x22, 0xF9, 0xE2, 0x4A, 0xBC, 0x35, 0xCA,
    0xEE, 0x78, 0x05, 0x6B, 0x51, 0xE1, 0x59, 0xA3, 0xF2, 0x71, 0x56, 0x11,
    0x6A, 0x89, 0x94, 0x65, 0x8C, 0xBB, 0x77, 0x3C, 0x7B, 0x28, 0xAB, 0xD2,
    0x31, 0xDE, 0xC4, 0x5F, 0xCC, 0xCF, 0x76, 0x2C, 0xB8, 0xD8, 0x2E, 0x36,
    0xDB, 0x69, 0xB3, 0x14, 0x95, 0xBE, 0x62, 0xA1, 0x3B, 0x16, 0x66, 0xE9,
    0x5C, 0x6C, 0x6D, 0xAD, 0x37, 0x61, 0x4B, 0xB9, 0xE3, 0xBA, 0xF1, 0xA0,
    0x85, 0x83, 0xDA, 0x47, 0xC5, 0xB0, 0x33, 0xFA, 0x96, 0x6F, 0x6E, 0xC2,
    0xF6, 0x50, 0xFF, 0x5D, 0xA9, 0x8E, 0x17, 0x1B, 0x97, 0x7D, 0xEC, 0x58,
    0xF7, 0x1F, 0xFB, 0x7C, 0x09, 0x0D, 0x7A, 0x67, 0x45, 0x87, 0xDC, 0xE8,
    0x4F, 0x1D, 0x4E, 0x04, 0xEB, 0xF8, 0xF3, 0x3E, 0x3D, 0xBD, 0x8A, 0x88,
    0xDD, 0xCD, 0x0B, 0x13, 0x98, 0x02, 0x93, 0x80, 0x90, 0xD0, 0x24, 0x34,
    0xCB, 0xED, 0xF4, 0xCE, 0x99, 0x10, 0x44, 0x40, 0x92, 0x3A, 0x01, 0x26,
    0x12, 0x1A, 0x48, 0x68, 0xF5, 0x81, 0x8B, 0xC7, 0xD6, 0x20, 0x0A, 0x08,
    0x00, 0x4C, 0xD7, 0x74
]

LIN_VEC = [
    1, 148, 32, 133, 16, 194, 192, 1, 251, 1, 192, 194, 16, 133, 32, 148
]


class CryptAlgorithms:

    def __init__(self):
        try:
            self.GALOIS = np.loadtxt("galois.txt")
        except Exception as e:
            np.savetxt('galois.txt',
                       [[CryptAlgorithms.FAIRM(i, j) for j in range(256)]
                        for i in range(256)])
        self.GALOIS = np.loadtxt("galois.txt")

    @staticmethod
    def FAIRM(a, b):
        power2 = [
            1, 2, 4, 8, 16, 32, 64, 128, 195, 69, 138, 215, 109, 218, 119, 238,
            31, 62, 124, 248, 51, 102, 204, 91, 182, 175, 157, 249, 49, 98,
            196, 75, 150, 239, 29, 58, 116, 232, 19, 38, 76, 152, 243, 37, 74,
            148, 235, 21, 42, 84, 168, 147, 229, 9, 18, 36, 72, 144, 227, 5,
            10, 20, 40, 80, 160, 131, 197, 73, 146, 231, 13, 26, 52, 104, 208,
            99, 198, 79, 158, 255, 61, 122, 244, 43, 86, 172, 155, 245, 41, 82,
            164, 139, 213, 105, 210, 103, 206, 95, 190, 191, 189, 185, 177,
            161, 129, 193, 65, 130, 199, 77, 154, 247, 45, 90, 180, 171, 149,
            233, 17, 34, 68, 136, 211, 101, 202, 87, 174, 159, 253, 57, 114,
            228, 11, 22, 44, 88, 176, 163, 133, 201, 81, 162, 135, 205, 89,
            178, 167, 141, 217, 113, 226, 7, 14, 28, 56, 112, 224, 3, 6, 12,
            24, 48, 96, 192, 67, 134, 207, 93, 186, 183, 173, 153, 241, 33, 66,
            132, 203, 85, 170, 151, 237, 25, 50, 100, 200, 83, 166, 143, 221,
            121, 242, 39, 78, 156, 251, 53, 106, 212, 107, 214, 111, 222, 127,
            254, 63, 126, 252, 59, 118, 236, 27, 54, 108, 216, 115, 230, 15,
            30, 60, 120, 240, 35, 70, 140, 219, 117, 234, 23, 46, 92, 184, 179,
            165, 137, 209, 97, 194, 71, 142, 223, 125, 250, 55, 110, 220, 123,
            246, 47, 94, 188, 187, 181, 169, 145, 225, 1
        ]
        exponent = [
            0, 255, 1, 157, 2, 59, 158, 151, 3, 53, 60, 132, 159, 70, 152, 216,
            4, 118, 54, 38, 61, 47, 133, 227, 160, 181, 71, 210, 153, 34, 217,
            16, 5, 173, 119, 221, 55, 43, 39, 191, 62, 88, 48, 83, 134, 112,
            228, 247, 161, 28, 182, 20, 72, 195, 211, 242, 154, 129, 35, 207,
            218, 80, 17, 204, 6, 106, 174, 164, 120, 9, 222, 237, 56, 67, 44,
            31, 40, 109, 192, 77, 63, 140, 89, 185, 49, 177, 84, 125, 135, 144,
            113, 23, 229, 167, 248, 97, 162, 235, 29, 75, 183, 123, 21, 95, 73,
            93, 196, 198, 212, 12, 243, 200, 155, 149, 130, 214, 36, 225, 208,
            14, 219, 189, 81, 245, 18, 240, 205, 202, 7, 104, 107, 65, 175,
            138, 165, 142, 121, 233, 10, 91, 223, 147, 238, 187, 57, 253, 68,
            51, 45, 116, 32, 179, 41, 171, 110, 86, 193, 26, 78, 127, 64, 103,
            141, 137, 90, 232, 186, 146, 50, 252, 178, 115, 85, 170, 126, 25,
            136, 102, 145, 231, 114, 251, 24, 169, 230, 101, 168, 250, 249,
            100, 98, 99, 163, 105, 236, 8, 30, 66, 76, 108, 184, 139, 124, 176,
            22, 143, 96, 166, 74, 234, 94, 122, 197, 92, 199, 11, 213, 148, 13,
            224, 244, 188, 201, 239, 156, 254, 150, 58, 131, 52, 215, 69, 37,
            117, 226, 46, 209, 180, 15, 33, 220, 172, 190, 42, 82, 87, 246,
            111, 19, 27, 241, 194, 206, 128, 203, 79
        ]
        if a == 0 or b == 0:
            return 0
        a_pow = exponent[a]
        b_pow = exponent[b]
        return power2[(a_pow + b_pow) % 255]

    def M(self, a, b):
        return int(self.GALOIS[a][b])

    @staticmethod
    def DS(x):
        return list(map(lambda b: RESBOX[b], x))

    @staticmethod
    def S(x):
        return list(map(lambda b: SBOX[b], x))

    def DL(self, x):
        x = deque(x)
        for _ in range(16):
            c = reduce(lambda a, b: a ^ b,
                       [self.M(x[i], LIN_VEC[i]) for i in range(len(LIN_VEC))])
            x.rotate(-1)
            x[15] = c
        return list(x)

    def L(self, x):
        x = deque(x)
        for _ in range(16):
            c = reduce(lambda a, b: a ^ b,
                       [self.M(x[i], LIN_VEC[i + 1])
                        for i in range(15)]) ^ x[15]
            x.rotate(1)
            x[0] = c
        return list(x)

    @staticmethod
    def X(a, b):
        return list(map(lambda x: x[0] ^ x[1], zip(a, b)))

    @staticmethod
    def constantToBytes(x, size):
        return [(x >> i * 8) & 0xff for i in range(size)][::-1]

    @staticmethod
    def bytesToString(b):
        return "".join(list(map(lambda x: "%02x" % x, b)))

    def feistelNetwork(self, l, incriment):
        constants = [
            0x6ea276726c487ab85d27bd10dd849401,
            0xdc87ece4d890f4b3ba4eb92079cbeb02,
            0xb2259a96b4d88e0be7690430a44f7f03,
            0x7bcd1b0b73e32ba5b79cb140f2551504,
            0x156f6d791fab511deabb0c502fd18105,
            0xa74af7efab73df160dd208608b9efe06,
            0xc9e8819dc73ba5ae50f5b570561a6a07,
            0xf6593616e6055689adfba18027aa2a08,
            0x98fb40648a4d2c31f0dc1c90fa2ebe09,
            0x2adedaf23e95a23a17b518a05e61c10a,
            0x447cac8052ddd8824a92a5b083e5550b,
            0x8d942d1d95e67d2c1a6710c0d5ff3f0c,
            0xe3365b6ff9ae07944740add0087bab0d,
            0x5113c1f94d76899fa029a9e0ac34d40e,
            0x3fb1b78b213ef327fd0e14f071b0400f,
            0x2fb26c2c0f0aacd1993581c34e975410,
            0x41101a5e6342d669c4123cd39313c011,
            0xf33580c8d79a5862237b38e3375cbf12,
            0x9d97f6babbd222da7e5c85f3ead82b13,
            0x547f77277ce987742ea93083bcc24114,
            0x3add015510a1fdcc738e8d936146d515,
            0x88f89bc3a47973c794e789a3c509aa16,
            0xe65aedb1c831097fc9c034b3188d3e17,
            0xd9eb5a3ae90ffa5834ce2043693d7e18,
            0xb7492c48854780e069e99d53b4b9ea19,
            0x056cb6de319f0eeb8e80996310f6951a,
            0x6bcec0ac5dd77453d3a72473cd72011b,
            0xa22641319aecd1fd835291039b686b1c,
            0xcc843743f6a4ab45de752c1346ecff1d,
            0x7ea1add5427c254e391c2823e2a3801e,
            0x1003dba72e345ff6643b95333f27141f,
            0x5ea7d8581e149b61f16ac1459ceda820
        ]
        constants = list(
            map(lambda x: CryptAlgorithms.constantToBytes(x, 16), constants))
        for i in range(1, 9):
            l = CryptAlgorithms.X(
                self.L(
                    CryptAlgorithms.S(
                        CryptAlgorithms.X(l[:16],
                                          constants[i - 1 + incriment]))),
                l[16:]) + l[:16]
        return l

    def keygen(self, x):
        keys = []
        keys.append(x[:16])
        keys.append(x[16:])
        incriment = 0

        for _ in range(4):
            x = self.feistelNetwork(x, incriment)
            keys.append(x[:16])
            keys.append(x[16:])
            incriment += 8
        return keys

    def cipher(self, block, keys):
        for i in range(9):
            block = self.L(CryptAlgorithms.S(CryptAlgorithms.X(keys[i],
                                                               block)))
        return CryptAlgorithms.X(block, keys[9])

    def decipher(self, block, keys):
        block = CryptAlgorithms.X(block, keys[9])
        for i in range(1, 10):
            block = CryptAlgorithms.X(CryptAlgorithms.DS(self.DL(block)),
                                      keys[9 - i])
        return block


class CryptingError(Exception):
    pass


class KuznechikACPKM:

    def __init__(self, key):
        if key > 1 << 128:
            raise CryptingError('Invalid key lenght, should be 32 bytes max')
        self.c = CryptAlgorithms()
        self.keys = self.c.keygen(CryptAlgorithms.constantToBytes(key, 32))

    def _cryptline(self, line):
        hexline = list(line.encode('utf-8'))
        padlen = 16 - len(hexline) % 16
        padzero = 16 - len(hex(padlen)[2:])
        print(hexline)
        hexline = hexline + [0] * (padlen + padzero) + [padlen]
        print(hexline)
        for i in range(len(hexline)//16):

            hexline[i * 16:i * 16 + 16] = self.c.cipher(hexline[i * 16:i * 16 + 16], self.keys)

        return CryptAlgorithms.bytesToString(hexline)

    def _decryptline(self, line):
        hexline = list(binascii.unhexlify(line))
        for i in range(len(hexline)//16):
            hexline[i * 16:i * 16 + 16] = self.c.decipher(hexline[i * 16:i * 16 + 16], self.keys)
        padlen = hexline[-1]
        return "".join(list(map(chr, hexline[:-padlen - 16])))

    def crypt_file(self, path):
        if not os.path.exists(path):
            raise CryptingError('Path is invalid')
        o = open(path, 'r')
        w = open(f'./crypted-{path.split("/")[-1]}.txt', 'a')
        for line in o.readlines():
            w.write(self._cryptline(line))
        o.close()
        w.close()

    def decrypt_file(self, path):
        o = open(path, 'r')
        w = open(f'./decrypted-{path.split("/")[-1]}.txt', 'a')
        for line in o.readlines():
            w.write(self._decryptline(line))
        o.close()
        w.close()


if __name__ == "__main__":
#Test
    c = KuznechikACPKM(0)
    c.crypt_file("hw.txt")
    c.decrypt_file("crypted-hw.txt.txt")
