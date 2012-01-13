
import numpy as np
import numpy.lib.format as npfor

def read_header_data(fname):

    fp = open(fname, 'r')
    version = npfor.read_magic(fp)
    if version != (1, 0):
        msg = "only support version (1,0) of file format, not %r"
        raise ValueError(msg % (version,))
    shape, fortran_order, dtype = npfor.read_array_header_1_0(fp)
    header_length = fp.tell()
    return shape, fortran_order, dtype, header_length

def write_header_data(fname, header_data):
    
    version = (1, 0)
    fp = open(fname, 'r+')
    fp.write(npfor.magic(*version))
    # Custom version fo write_array_header, not the np.lib.format version.
    write_array_header_1_0(fp, header_data)

def pack_header_data(shape, fortran_order, dtype):
    
    # Do very strict type checking, which is normally a not done in python.
    # We need repr() to work perfectly.
    msg = "`shape` must me a tuple of intergers."
    if type(shape) != type(()):
        raise TypeError(msg)
    for s in shape:
        if type(s) != type(1):
            raise TypeError(msg)
    if type(fortran_order) != type(True):
        msg = "`fortran_order` must be boolian."
        raise TypeError(msg)

    header_data = {}
    header_data['shape'] = shape
    header_data['fortran_order'] = fortran_order
    header_data['descr'] = npfor.dtype_to_descr(np.dtype(dtype))
    return header_data

def get_header_length(header):
    """Gets the total length of the header given the header data.
    
    Works for header data in a dictionary (as returned by `pack_header_data` or
    for data already packed into a string (as returned by `get_header_string`).
    """
    
    if isinstance(header, dict):
        header = get_header_str(header)
    return npfor.MAGIC_LEN + 2 + len(header) 

def get_header_str(header_data):
    
    header = ["{"]
    for key, value in sorted(header_data.items()):
        # Need to use repr here, since we eval these when reading
        header.append("'%s': %s, " % (key, repr(value)))
    header.append("}")
    header = "".join(header)
    # Pad the header with spaces and a final newline such that the magic
    # string, the header-length short and the header are aligned on a 4096-byte
    # boundary.  Hopefully, some system, possibly memory-mapping, can take
    # advantage of our premature optimization.
    # 1 for the newline
    current_header_len = get_header_length(header) + 1  # 1 for newline.  
    topad = 4096 - (current_header_len % 4096)
    header = '%s%s\n' % (header, ' '*topad)
    return header

def write_array_header_1_0(fp, d):
    """ Write the header for an array using the 1.0 format.

    This version of write array header has been modified to align the start of
    the array data with the 4096 bytes, corresponding to the page size of most
    systems.  This is so the npy files can be easily memmaped.

    Parameters
    ----------
    fp : filelike object
    d : dict
        This has the appropriate entries for writing its string representation
        to the header of the file.
    """

    import struct
    header = get_header_str(d)
    if len(header) >= (256*256):
        raise ValueError("header does not fit inside %s bytes" % (256*256))
    header_len_str = struct.pack('<H', len(header))
    fp.write(header_len_str)
    fp.write(header)

