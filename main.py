import numpy as np
import os
from functools import reduce
from collections import deque
import binascii

class crypt_methods:
    def __init__(self):
        self.a = 0
        self.GALOIS = np.loadtxt('galois.txt')
    def FAIRM(self, a, b):
        power2 = [ 1, 2, 4, 8, 16, 32, 64, 128, 195, 69, 138, 215, 109, 218, 119, 238,31, 62, 124, 248, 51, 102, 204, 91, 182, 175, 157, 249, 49, 98, 196, 75,150, 239, 29, 58, 116, 232, 19, 38, 76, 152, 243, 37, 74, 148, 235, 21,42, 84, 168, 147, 229, 9, 18, 36, 72, 144, 227, 5, 10, 20, 40, 80,160, 131, 197, 73, 146, 231, 13, 26, 52, 104, 208, 99, 198, 79, 158, 255,61, 122, 244, 43, 86, 172, 155, 245, 41, 82, 164, 139, 213, 105, 210, 103,206, 95, 190, 191, 189, 185, 177, 161, 129, 193, 65, 130, 199, 77, 154, 247,45, 90, 180, 171, 149, 233, 17, 34, 68, 136, 211, 101, 202, 87, 174, 159,253, 57, 114, 228, 11, 22, 44, 88, 176, 163, 133, 201, 81, 162, 135, 205,89, 178, 167, 141, 217, 113, 226, 7, 14, 28, 56, 112, 224, 3, 6, 12,24, 48, 96, 192, 67, 134, 207, 93, 186, 183, 173, 153, 241, 33, 66, 132,203, 85, 170, 151, 237, 25, 50, 100, 200, 83, 166, 143, 221, 121, 242, 39,78, 156, 251, 53, 106, 212, 107, 214, 111, 222, 127, 254, 63, 126, 252, 59,118, 236, 27, 54, 108, 216, 115, 230, 15, 30, 60, 120, 240, 35, 70, 140,219, 117, 234, 23, 46, 92, 184, 179, 165, 137, 209, 97, 194, 71, 142, 223,125, 250, 55, 110, 220, 123, 246, 47, 94, 188, 187, 181, 169, 145, 225, 1 ]
        exponent = [ 0, 255, 1, 157, 2, 59, 158, 151, 3, 53, 60, 132, 159, 70, 152, 216,4, 118, 54, 38, 61, 47, 133, 227, 160, 181, 71, 210, 153, 34, 217, 16,5, 173, 119, 221, 55, 43, 39, 191, 62, 88, 48, 83, 134, 112, 228, 247,161, 28, 182, 20, 72, 195, 211, 242, 154, 129, 35, 207, 218, 80, 17, 204,6, 106, 174, 164, 120, 9, 222, 237, 56, 67, 44, 31, 40, 109, 192, 77,63, 140, 89, 185, 49, 177, 84, 125, 135, 144, 113, 23, 229, 167, 248, 97,162, 235, 29, 75, 183, 123, 21, 95, 73, 93, 196, 198, 212, 12, 243, 200,155, 149, 130, 214, 36, 225, 208, 14, 219, 189, 81, 245, 18, 240, 205, 202,7, 104, 107, 65, 175, 138, 165, 142, 121, 233, 10, 91, 223, 147, 238, 187,57, 253, 68, 51, 45, 116, 32, 179, 41, 171, 110, 86, 193, 26, 78, 127,64, 103, 141, 137, 90, 232, 186, 146, 50, 252, 178, 115, 85, 170, 126, 25,136, 102, 145, 231, 114, 251, 24, 169, 230, 101, 168, 250, 249, 100, 98, 99,163, 105, 236, 8, 30, 66, 76, 108, 184, 139, 124, 176, 22, 143, 96, 166,74, 234, 94, 122, 197, 92, 199, 11, 213, 148, 13, 224, 244, 188, 201, 239,156, 254, 150, 58, 131, 52, 215, 69, 37, 117, 226, 46, 209, 180, 15, 33,220, 172, 190, 42, 82, 87, 246, 111, 19, 27, 241, 194, 206, 128, 203, 79 ]
        if a == 0 or b == 0:
            return 0
        a_pow = exponent[a]
        b_pow = exponent[b]
        return power2[(a_pow + b_pow)%255]
    def _M(self, a, b):
        return int(self.GALOIS[a][b])
    def _DS(self, x):
        depihex = ['a5', '2d', '32', '8f', '0e', '30', '38', 'c0', '54', 'e6', '9e', '39', '55', '7e', '52', '91', '64', '03', '57', '5a', '1c', '60', '07', '18', '21', '72', 'a8', 'd1', '29', 'c6', 'a4', '3f', 'e0', '27', '8d', '0c', '82', 'ea', 'ae', 'b4', '9a', '63', '49', 'e5', '42', 'e4', '15', 'b7', 'c8', '06', '70', '9d', '41', '75', '19', 'c9', 'aa', 'fc', '4d', 'bf', '2a', '73', '84', 'd5', 'c3', 'af', '2b', '86', 'a7', 'b1', 'b2', '5b', '46', 'd3', '9f', 'fd', 'd4', '0f', '9c', '2f', '9b', '43', 'ef', 'd9', '79', 'b6', '53', '7f', 'c1', 'f0', '23', 'e7', '25', '5e', 'b5', '1e', 'a2', 'df', 'a6', 'fe', 'ac', '22', 'f9', 'e2', '4a', 'bc', '35', 'ca', 'ee', '78', '05', '6b', '51', 'e1', '59', 'a3', 'f2', '71', '56', '11', '6a', '89', '94', '65', '8c', 'bb', '77', '3c', '7b', '28', 'ab', 'd2', '31', 'de', 'c4', '5f', 'cc', 'cf', '76', '2c', 'b8', 'd8', '2e', '36', 'db', '69', 'b3', '14', '95', 'be', '62', 'a1', '3b', '16', '66', 'e9', '5c', '6c', '6d', 'ad', '37', '61', '4b', 'b9', 'e3', 'ba', 'f1', 'a0', '85', '83', 'da', '47', 'c5', 'b0', '33', 'fa', '96', '6f', '6e', 'c2', 'f6', '50', 'ff', '5d', 'a9', '8e', '17', '1b', '97', '7d', 'ec', '58', 'f7', '1f', 'fb', '7c', '09', '0d', '7a', '67', '45', '87', 'dc', 'e8', '4f', '1d', '4e', '04', 'eb', 'f8', 'f3', '3e', '3d', 'bd', '8a', '88', 'dd', 'cd', '0b', '13', '98', '02', '93', '80', '90', 'd0', '24', '34', 'cb', 'ed', 'f4', 'ce', '99', '10', '44', '40', '92', '3a', '01', '26', '12', '1a', '48', '68', 'f5', '81', '8b', 'c7', 'd6', '20', '0a', '08', '00', '4c', 'd7', '74']
        return ''.join([depihex[int(x[i * 2:i * 2 + 2], 16)] for i in range(int(len(x)/2))])
    def _S(self, x):#С-бокс
        pihex = ['fc', 'ee', 'dd', '11', 'cf', '6e', '31', '16', 'fb', 'c4', 'fa', 'da', '23', 'c5', '04', '4d', 'e9', '77', 'f0', 'db', '93', '2e', '99', 'ba', '17', '36', 'f1', 'bb', '14', 'cd', '5f', 'c1', 'f9', '18', '65', '5a', 'e2', '5c', 'ef', '21', '81', '1c', '3c', '42', '8b', '01', '8e', '4f', '05', '84', '02', 'ae', 'e3', '6a', '8f', 'a0', '06', '0b', 'ed', '98', '7f', 'd4', 'd3', '1f', 'eb', '34', '2c', '51', 'ea', 'c8', '48', 'ab', 'f2', '2a', '68', 'a2', 'fd', '3a', 'ce', 'cc', 'b5', '70', '0e', '56', '08', '0c', '76', '12', 'bf', '72', '13', '47', '9c', 'b7', '5d', '87', '15', 'a1', '96', '29', '10', '7b', '9a', 'c7', 'f3', '91', '78', '6f', '9d', '9e', 'b2', 'b1', '32', '75', '19', '3d', 'ff', '35', '8a', '7e', '6d', '54', 'c6', '80', 'c3', 'bd', '0d', '57', 'df', 'f5', '24', 'a9', '3e', 'a8', '43', 'c9', 'd7', '79', 'd6', 'f6', '7c', '22', 'b9', '03', 'e0', '0f', 'ec', 'de', '7a', '94', 'b0', 'bc', 'dc', 'e8', '28', '50', '4e', '33', '0a', '4a', 'a7', '97', '60', '73', '1e', '00', '62', '44', '1a', 'b8', '38', '82', '64', '9f', '26', '41', 'ad', '45', '46', '92', '27', '5e', '55', '2f', '8c', 'a3', 'a5', '7d', '69', 'd5', '95', '3b', '07', '58', 'b3', '40', '86', 'ac', '1d', 'f7', '30', '37', '6b', 'e4', '88', 'd9', 'e7', '89', 'e1', '1b', '83', '49', '4c', '3f', 'f8', 'fe', '8d', '53', 'aa', '90', 'ca', 'd8', '85', '61', '20', '71', '67', 'a4', '2d', '2b', '09', '5b', 'cb', '9b', '25', 'd0', 'be', 'e5', '6c', '52', '59', 'a6', '74', 'd2', 'e6', 'f4', 'b4', 'c0', 'd1', '66', 'af', 'c2', '39', '4b', '63', 'b6']
        return ''.join([pihex[int(x[i * 2:i * 2 + 2], 16)] for i in range(int(len(x)/2))])
    def _DL(self, x):
        constants = [1, 148, 32, 133, 16, 194, 192, 1, 251, 1, 192, 194, 16, 133, 32, 148]
        x = [int(x[i * 2: i * 2 + 2], 16) for i in range(int(len(x)/2))]
        
        for _ in range(16):
            c = reduce(lambda a, b: a ^ b, map(int, [self._M(x[i], constants[i]) for i in range(len(constants))]))
            x = deque(x)
            x.rotate(-1)
            x[15] = c
        return ''.join([hex(pair)[2:] if len(hex(pair)[2:]) > 1 else '0'+hex(pair)[2:] for pair in x])
    def _L(self, x):
        constants = [1, 148, 32, 133, 16, 194, 192, 1, 251, 1, 192, 194, 16, 133, 32, 148]
        x = [int(x[i * 2: i * 2 + 2], 16) for i in range(int(len(x)/2))]
        for _ in range(16):
            c = reduce(lambda a, b: a ^ b, map(int, [self._M(x[i], constants[i + 1]) for i in range(15)])) ^ x[15]
            x = deque(x)
            x.rotate(1)
            x[0] = c

        return ''.join([hex(pair)[2:] if len(hex(pair)[2:]) > 1 else '0'+hex(pair)[2:] for pair in x])
    def _X(self, s1, s2):
        s1 = [int(s1[i * 2 : i * 2 + 2], 16) for i in range(int(len(s1) / 2))]
        s2 = [int(s2[i * 2 : i * 2 + 2], 16) for i in range(int(len(s2) / 2))]
        return ''.join([hex(s1[i] ^ s2[i])[2:] if len(hex(s1[i] ^ s2[i])[2:]) > 1 else '0'+hex(s1[i] ^ s2[i])[2:] for i in range(len(s1))])
    def _feistel(self, l,incrim): 
        #constants=['019484dd10bd275db87a486c7276a26e', '02ebcb7920b94ebab3f490d8e4ec87dc', '037f4fa4300469e70b8ed8b4969a25b2', '041555f240b19cb7a52be3730b1bcd7b', '0581d12f500cbbea1d51ab1f796d6f15', '06fe9e8b6008d20d16df73abeff74aa7', '076a1a5670b5f550aea53bc79d81e8c9', '082aaa2780a1fbad895605e6163659f6', '09be2efa901cdcf0312c4d8a6440fb98', '0ac1615ea018b5173aa2953ef2dade2a', '0b55e583b0a5924a82d8dd5280ac7c44', '0c3fffd5c010671a2c7de6951d2d948d', '0dab7b08d0ad40479407aef96f5b36e3', '0ed434ace0a929a09f89764df9c11351', '0f40b071f0140efd27f33e218bb7b13f', '1054974ec3813599d1ac0a0f2c6cb22f', '11c01393d33c12c469d642635e1a1041', '12bf5c37e3387b2362589ad7c88035f3', '132bd8eaf3855c7eda22d2bbbaf6979d', '1441c2bc8330a92e7487e97c27777f54', '15d54661938d8e73ccfda1105501dd3a', '16aa09c5a389e794c77379a4c39bf888', '173e8d18b334c0c97f0931c8b1ed5ae6', '187e3d694320ce3458fa0fe93a5aebd9', '19eab9b4539de969e0804785482c49b7', '1a95f6106399808eeb0e9f31deb66c05', '1b0172cd7324a7d35374d75dacc0ce6b', '1c6b689b03915283fdd1ec9a314126a2', '1dffec46132c75de45aba4f6433784cc', '1e80a3e223281c394e257c42d5ada17e', '1f14273f33953b64f65f342ea7db0310', '20a8ed9c45c16af1619b141e58d8a75e']
        constants=['6ea276726c487ab85d27bd10dd849401', 'dc87ece4d890f4b3ba4eb92079cbeb02', 'b2259a96b4d88e0be7690430a44f7f03', '7bcd1b0b73e32ba5b79cb140f2551504', '156f6d791fab511deabb0c502fd18105', 'a74af7efab73df160dd208608b9efe06', 'c9e8819dc73ba5ae50f5b570561a6a07', 'f6593616e6055689adfba18027aa2a08', '98fb40648a4d2c31f0dc1c90fa2ebe09', '2adedaf23e95a23a17b518a05e61c10a', '447cac8052ddd8824a92a5b083e5550b', '8d942d1d95e67d2c1a6710c0d5ff3f0c', 'e3365b6ff9ae07944740add0087bab0d', '5113c1f94d76899fa029a9e0ac34d40e', '3fb1b78b213ef327fd0e14f071b0400f', '2fb26c2c0f0aacd1993581c34e975410', '41101a5e6342d669c4123cd39313c011', 'f33580c8d79a5862237b38e3375cbf12', '9d97f6babbd222da7e5c85f3ead82b13', '547f77277ce987742ea93083bcc24114', '3add015510a1fdcc738e8d936146d515', '88f89bc3a47973c794e789a3c509aa16', 'e65aedb1c831097fc9c034b3188d3e17', 'd9eb5a3ae90ffa5834ce2043693d7e18', 'b7492c48854780e069e99d53b4b9ea19', '056cb6de319f0eeb8e80996310f6951a', '6bcec0ac5dd77453d3a72473cd72011b', 'a22641319aecd1fd835291039b686b1c', 'cc843743f6a4ab45de752c1346ecff1d', '7ea1add5427c254e391c2823e2a3801e', '1003dba72e345ff6643b95333f27141f', '5ea7d8581e149b61f16ac1459ceda820']
        for i in range (1, 9):
            l= self._X(self._L(self._S(self._X(l[:(int(len(l)/2))],constants[i-1+incrim]))), l[(int(len(l)/2)):])+l[:(int(len(l)/2))]
        return l
    def keygen(self, x): 
        #constants=['019484dd10bd275db87a486c7276a26e', '02ebcb7920b94ebab3f490d8e4ec87dc', '037f4fa4300469e70b8ed8b4969a25b2', '041555f240b19cb7a52be3730b1bcd7b', '0581d12f500cbbea1d51ab1f796d6f15', '06fe9e8b6008d20d16df73abeff74aa7', '076a1a5670b5f550aea53bc79d81e8c9', '082aaa2780a1fbad895605e6163659f6', '09be2efa901cdcf0312c4d8a6440fb98', '0ac1615ea018b5173aa2953ef2dade2a', '0b55e583b0a5924a82d8dd5280ac7c44', '0c3fffd5c010671a2c7de6951d2d948d', '0dab7b08d0ad40479407aef96f5b36e3', '0ed434ace0a929a09f89764df9c11351', '0f40b071f0140efd27f33e218bb7b13f', '1054974ec3813599d1ac0a0f2c6cb22f', '11c01393d33c12c469d642635e1a1041', '12bf5c37e3387b2362589ad7c88035f3', '132bd8eaf3855c7eda22d2bbbaf6979d', '1441c2bc8330a92e7487e97c27777f54', '15d54661938d8e73ccfda1105501dd3a', '16aa09c5a389e794c77379a4c39bf888', '173e8d18b334c0c97f0931c8b1ed5ae6', '187e3d694320ce3458fa0fe93a5aebd9', '19eab9b4539de969e0804785482c49b7', '1a95f6106399808eeb0e9f31deb66c05', '1b0172cd7324a7d35374d75dacc0ce6b', '1c6b689b03915283fdd1ec9a314126a2', '1dffec46132c75de45aba4f6433784cc', '1e80a3e223281c394e257c42d5ada17e', '1f14273f33953b64f65f342ea7db0310', '20a8ed9c45c16af1619b141e58d8a75e']
        keys=[]
        keys.append(x[:(int(len(x)/2))])
        keys.append(x[(int(len(x)/2)):])
        incrim = 0
        for _ in range (4):
            x=self._feistel(x,incrim)
            keys.append(x[:(int(len(x)/2))])
            keys.append(x[(int(len(x)/2)):])
            incrim += 8
        return keys
    def cipher(self, block, keys):
        for i in range(9):
            block = self._L(self._S(self._X(keys[i], block)))
        return self._X(block, keys[9])
    def decipher(self, block,keys):
        block = self._X(block,keys[9])
        for i in range(1, 10):
            block = self._X(self._DS(self._DL(block)),keys[9-i])
        return block

class CryptingError(Exception):
    pass


class crypter:
    def __init__(self, path, key):
        if not os.path.exists(path):
            raise CryptingError('Path is invalid')
        try:
            int(key, 16)
        except ValueError:
            raise CryptingError('Key should be in a hexademical format')
        if len(key) != 64:
            raise CryptingError('Invalid key lenght, should be 64')
        self.c = crypt_methods()
        self.path = path
        self.keys = self.c.keygen(key)
    
        
    def _cryptline(self, line):
        hexline = str(binascii.hexlify(line.encode('utf-8')))[2:-1]
        padlen =32 - len(hexline) % 32
        padzero = 32 - len(hex(padlen)[2:])
        hexline = hexline + '0' * (padlen + padzero) + hex(padlen)[2:] 
        return ''.join(map(lambda x: self.c.cipher(x, self.keys), [hexline[i:i+32] for i in range(0, len(hexline), 32)]))
    def _decryptline(self, line):
        line = line[:-1]
        hexline = ''.join(map(lambda x: self.c.decipher(x, self.keys), [line[i:i+32] for i in range(0, len(line), 32)]))
        padlen = int(hexline[-32:], 16)
        return str(binascii.unhexlify(hexline[:-32  - padlen]), 'utf-8')
    def crypt_file(self):
        o = open(self.path, 'r')
        w = open('./crypted.txt', 'a')
        for line in o.readlines():
            w.write(self._cryptline(line) + "\n")
        o.close()
        w.close()
    def decrypt_file(self):
        o = open(self.path, 'r')
        w = open('./decrypted.txt', 'a')
        for line in o.readlines():
            w.write(self._decryptline(line))
        o.close()
        w.close()

#np.savetxt('galois.txt',[[c.FAIRM(i, j)for j in range(256)]for i in range(256)])