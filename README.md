# ACPKM-kuznechik

This is an ACPKM - Kuznechik implementation according to "GOST 2015" standart [GOST 2015.pdf](https://github.com/forentfraps/ACPKM-kuznechik/files/9097518/GOST.2015.pdf) (Its in russian, because it was developed there)

## Installation
  - Clone the repo
  - ```pip install -r -requirements.txt```



## Usage

Initialise crypter() with a target file path and a 64 lengh HEX key


### crypt_file()

Encrypts the target file into ./crypted.txt


### decrypt_file()

Decrypt the target file into ./decrypted.txt




## Documentation and some explainations

Main cryptographics methods are located in crypt_methods, I will briefly explain whats their function, however, its done much better in the GOST


### FAIRM()

It is multiplication in galois field [GF(2^8)], basically multiplying polynomials, this is semi-precalculated implementation

### _M()

Because we are only multiplying numbers from 0x00 to 0xff (from 0 to 255), we are working with a very small number of possible outputs 255^2, thus by precalculating all the results, we end up saving precious computing time


### _S()

Is and so-called S-BOX, basically just a bijection table (present in GOST document, its called pi there). The whole reason behind its presence, is to remove linearity, thus exponentially increasing the difficulty to crack the algorithm


### _L()

Basically we split the input block in to 16 bytes, then multiply with (1, 148, 32, 133, 16, 194, 192, 1, 251, 1, 192, 194, 16, 133, 32, 148), xor results, shift array, and put the xored result as the last element. Repeated 16 times


### _X()

Xoring 2 strings


### _DS() and _DL() 

These are the same methods as their non-D variants, but implemented backwards (used to decrypt the result)


### _feistel() and keygen()

These methods are used to derive child keys from master key, as you may have guessed the idea behind is to use the feistel network and previously implemented functions (_S, _L). Feistel network also uses some constants, which were secretly derived by some incredible mathematitians


### cipher() and decipher()

Ciphers and deciphers the block accordingly. For example if T is out block, then to cipher it we would L(S(X(T, K1), and repeat with each child key 


## Note

I will not comment the code :-)
