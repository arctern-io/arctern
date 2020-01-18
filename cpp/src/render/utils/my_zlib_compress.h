#include "stb/stb_image_write.h"
#include "miniz/miniz.h"

STBIWDEF unsigned char * my_zlib_compress(unsigned char *data, int data_len,    int *out_len, int quality){
    mz_ulong buflen = mz_compressBound(data_len);
    // Note that the returned buffer will be free'd by stbi_write_png*()
    // with STBIW_FREE()
    unsigned char* buf = (unsigned char*)malloc(buflen);
    if( buf == NULL
        || mz_compress2(buf, &buflen, data, data_len, quality) != 0 )
    {
        free(buf);
        return NULL;
    }
    *out_len = buflen;
    return buf;
}