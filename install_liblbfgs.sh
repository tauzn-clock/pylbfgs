cd liblbfgs
apt update
apt install -y libtool automake virtualenv
./autogen.sh
./configure --enable-sse2
make
make install