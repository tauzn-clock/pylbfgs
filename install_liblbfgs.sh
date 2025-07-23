cd liblbfgs
apt install libtool automake virtualenv
./autogen.sh
./configure --enable-sse2
make
make install