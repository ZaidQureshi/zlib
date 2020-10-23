#!/usr/bin/python
import sys
import zlib
import cv2
import os
import multiprocessing
import mmap
import zstandard as zstd
import mmap
import numpy as np

MIN_MATCH = 1;
#N_PROCS = 80-4;
N_PROCS = 1;
WARP_SIZE = 32;
PATH = "/home/zaid/.local/share/Steam/steamapps/common/dds/"
#PATH = "/mnt/707f56b3-4a3c-4769-aaea-6f3b497651dc/zaid/datasets/fin_textures/"
#PATH = "./imgs/"
#PATH = "/home/zaid/nfs/datasets/new_textures/"

def find_match(dict, input, input_offset):
    max_len = 0;
    max_off = 0;
    cur_len = 0;
    cur_off = 0;
    dict_len = len(dict);
    input_len = len(input);
    while(cur_off < len(dict)):

        if ((input_offset < input_len) and (input[input_offset] == dict[cur_off]).all()):
            cur_len = 1;

        while (((cur_off + cur_len) < dict_len) and ((input_offset + cur_len) < input_len) and (input[input_offset+cur_len] == dict[cur_off+cur_len]).all()):
            cur_len = cur_len + 1;

        if (cur_len > max_len):
            max_len = cur_len;
            max_off = cur_off;

        cur_off = cur_off + 1;

    return max_off,max_len;

def compress_texture(input, dict):
    pointers = [];
    match_len = 0;
    match_off = 0;
    prev_off = 0;
    prev_len = 0;
    #prev = false;
    i = 0;
    count = 0;
    input_len = len(input);
    while i < input_len:
        match_off,match_len = find_match(dict, input, i);
        if match_len < MIN_MATCH:
            dict.append(input[i]);

            #if (prev_len+prev_off) == len(dict):
            #    prev = true;
            #    pointers[-1][1] = pointers[-1][1] + 1;
            i = i + 1;
        else:
            i = i + match_len;
        if match_len == MIN_MATCH:
            count = count + 1;
        #if prev == false:
        pointers.append([match_off,match_len])
        #print(str(i));
        print(i)
        #prev_off = match_off;
        #prev_len = match_len;
        #prev = false;
        #print(i)
    return dict,pointers,count;

def merge_dict(fin_dict, dict):
    match_len = 0;
    match_off = 0;
    i = 0;
    dict_len = len(dict);
    while i < dict_len:
        match_off,match_len = find_match(fin_dict, dict, i);
        if match_len < MIN_MATCH:
            fin_dict.append(dict[i]);

            #if (prev_len+prev_off) == len(dict):
            #    prev = true;
            #    pointers[-1][1] = pointers[-1][1] + 1;
            i = i + 1;
        else:
            i = i + match_len;
        #if prev == false:

        #prev_off = match_off;
        #prev_len = match_len;
        #prev = false;
    return fin_dict;

def compress_dicts(dicts):
    fin_dict = dicts[0]

    for dict in dicts[1:]:
        if (dict != None):
            fin_dict = merge_dict(fin_dict, dict);
    return fin_dict;

#row major, single thread
def method1(input):
    dict = [];
    pointers = [];
    count = 0;
    input_reshaped = input.reshape(input.shape[0]*input.shape[1], -1);
    dict,pointers,count = compress_texture(input_reshaped, dict);
    #for i in range(input.shape[0]):
    #    pointers.append(None);
    #    dict,pointers[i],cnt = compress_texture(input[i], dict);
    #    count = count + cnt
    dict_size = len(dict);
    pointers_size = len(pointers);

    return [dict_size, pointers_size, count];
    #pointers_count = sum(map(len, pointers));

#row-major linear, warp 32 strided
def method2(input):
    input_reshaped = input.reshape(input.shape[0]*input.shape[1], -1);
    input_shfled = [None] * WARP_SIZE;
    dicts = [None] * WARP_SIZE;
    pointers = [None] * WARP_SIZE;
    counts = [0] * WARP_SIZE;
    for i in range(WARP_SIZE):
        input_shfled[i] = input_reshaped[i::WARP_SIZE];
        if (input_shfled[i].size == 0):
            break;
        dicts[i] = [];
        dicts[i],pointers[i],counts[i] = compress_texture(input_shfled[i],dicts[i]);
    ret = [sum(map(lambda x: len(x) if (x != None) else 0, dicts)), sum(map(lambda x: len(x) if (x != None) else 0,pointers)), sum(counts)];
    print("Before Merge Dicts\n")
    merged_dict = compress_dicts(dicts);
    ret.append(len(merged_dict));
    return ret;

#col-major linear, warp 32 strided
def method3(input):
    input_reshaped = input.transpose(1,0,2).reshape(input.shape[0]*input.shape[1], -1);
    input_shfled = [None] * WARP_SIZE;
    dicts = [None] * WARP_SIZE;
    pointers = [None] * WARP_SIZE;
    counts = [0] * WARP_SIZE;
    for i in range(WARP_SIZE):
        input_shfled[i] = input_reshaped[i::WARP_SIZE];
        if (input_shfled[i].size == 0):
            break;
        dicts[i] = [];
        dicts[i],pointers[i],counts[i] = compress_texture(input_shfled[i],dicts[i]);
    ret = [sum(map(lambda x: len(x) if (x != None) else 0, dicts)), sum(map(lambda x: len(x) if (x != None) else 0,pointers)), sum(counts)];
    merged_dict = compress_dicts(dicts);
    ret.append(len(merged_dict));
    return ret;

#row-major matrix, warp 32 strided (each thread gets rows)
def method4(input):
    dicts = [None] * WARP_SIZE;
    pointers = [None] * WARP_SIZE;
    counts = [0] * WARP_SIZE;
    inputs = [None] * WARP_SIZE;
    #input_reshaped = input.reshape(input.shape[0]*input.shape[1], -1);
    #dict,pointers,count = compress_texture(input_reshaped, dict);
    for i in range(WARP_SIZE):
        inputs[i] = input[i::WARP_SIZE];
        if (inputs[i].size == 0):
            break;
        inputs[i] = inputs[i].reshape(inputs[i].shape[0]*inputs[i].shape[1], -1);
        dicts[i] = [];
        dicts[i],pointers[i],counts[i] = compress_texture(inputs[i],dicts[i]);
    ret = [sum(map(lambda x: len(x) if (x != None) else 0, dicts)), sum(map(lambda x: len(x) if (x != None) else 0,pointers)), sum(counts)];
    merged_dict = compress_dicts(dicts);
    ret.append(len(merged_dict));
    return ret;

#col-major matrix, warp 32 strided (each thread gets cols)
def method5(input):
    dicts = [None] * WARP_SIZE;
    pointers = [None] * WARP_SIZE;
    counts = [0] * WARP_SIZE;
    inputs = [None] * WARP_SIZE;
    #input_reshaped = input.reshape(input.shape[0]*input.shape[1], -1);
    #dict,pointers,count = compress_texture(input_reshaped, dict);
    for i in range(WARP_SIZE):
        inputs[i] = input.transpose(1,0,2)[i::WARP_SIZE];
        if (inputs[i].size == 0):
            break;
        inputs[i] = inputs[i].reshape(inputs[i].shape[0]*inputs[i].shape[1], -1);
        dicts[i] = [];
        dicts[i],pointers[i],counts[i] = compress_texture(inputs[i],dicts[i]);
    ret = [sum(map(lambda x: len(x) if (x != None) else 0, dicts)), sum(map(lambda x: len(x) if (x != None) else 0,pointers)), sum(counts)];
    merged_dict = compress_dicts(dicts);
    ret.append(len(merged_dict));
    return ret;

def zlib_compress(data):
    zc = zlib.compress(data, level=9);
    return ["zlib size: " + str(len(zc))];

level = 22;
def zstd_compress(data):
    cctx = zstd.ZstdCompressor(level=level);
    zstdc = cctx.compress(data);
    return ["zstd size: " + str(len(zstdc))];


#strides = [4, 8, 16, 32];
#chunks = [4096, 32*1024, 64*1024];
strides = [32];
chunks = [128*1024];
def zlib_strided_compress(data):
    np_arr = np.frombuffer(data, dtype=np.uint8);
    data_copy = np.copy(np_arr);
    ret = [];

    for i in strides:
        c_count = 0;
        for k in chunks:
            l_data = len(data_copy);
            pad_needed = int(((int((l_data+k-1)/k)) * k) - l_data);

            #print(l_data)
            #print(pad_needed)
            nz = np.zeros(pad_needed, dtype=np.uint8);
            data_copy_n = np.concatenate((data_copy, nz));
            n_chunks = int(len(data_copy_n)/k);

            c_count = c_count + 1;
            t_l = 0;
            t_l_c = 0;
            t_l_c_d = 0;

            for p in range(n_chunks):
                sys.stderr.write("chunk: " + str(p) + " Start --------------------------------------------\n")
                #start = (0 if p == 0 else ((p)*k));
                start = p * k;
                end = (p+1) * k;
                #sys.stderr.write("start: " + str(start) + "\tend: " + str(end) + "\n")
                cur_data = data_copy_n[start:end];
                data_copy_r = cur_data.reshape((-1,i));
                input = [];
                input_compressed = []

                for j in range(WARP_SIZE):
                    sys.stderr.write("tid: " + str(p*WARP_SIZE+j) + "\tchunk: " + str(p) + "\tlane: " +str(j) + "\n");
                    sys.stderr.flush();
                    nd = (data_copy_r[j::WARP_SIZE]).reshape((-1)).tobytes();
                    input.append(nd);
                    #cctx = zstd.ZstdCompressor(level=level);
                    zc = zlib.compress(nd, level=9);
                    sys.stderr.flush();
                    #input_compressed.append(zstdc);
                    t_l_c = t_l_c + len(zc);
                    t_l = t_l + len(nd);
                sys.stderr.write("chunk: " + str(p) + " End --------------------------------------------\n")
            sys.stderr.write("zlib_" + str(i) + "_" + str(k) + " size: " + str(t_l_c) + "\tpadded size: " + str(len(data_copy_n)) + "\tt_l: " +str(t_l)+ "\torig size: " + str(len(data)) + "\n");
            ret.append("zlib_" + str(i) + "_" + str(k) + " size: " + str(t_l_c) + "\tpadded size: " + str(t_l) + "\torig size: " + str(len(data)) + "\n");


    return ret;


def zstd_strided_compress(data):
    np_arr = np.frombuffer(data, dtype=np.uint8);
    data_copy = np.copy(np_arr);
    ret = [];

    for i in strides:
        l_data = len(data_copy);
        pad_needed = int(((int((l_data+i-1)/i)) * i) - l_data);
        #print(l_data)
        #print(pad_needed)
        nz = np.zeros(pad_needed, dtype=np.uint8);
        data_copy_n = np.concatenate((data_copy, nz));
        #print(len(data_copy_n))
        data_copy_r = data_copy_n.reshape((-1,i));
        input = [];
        input_compressed = []
        t_l = 0;
        t_l_c = 0;
        t_l_c_d = 0;
        for j in range(WARP_SIZE):
            nd = (data_copy_r[j::WARP_SIZE]).reshape((-1)).tobytes();
            input.append(nd);
            cctx = zstd.ZstdCompressor(level=level);
            zstdc = cctx.compress(nd);
            #input_compressed.append(zstdc);
            t_l_c = t_l_c + len(zstdc);
            t_l = t_l + len(nd);
        ret.append("zstd_" + str(i) + " size: " + str(t_l_c));
        try:
            dict_data = zstd.train_dictionary(t_l, input, split_point=1.0, level=level);
            cctx = zstd.ZstdCompressor(dict_data=dict_data, level=level);
            for j in range(WARP_SIZE):
                cc = cctx.compress(input[j]);
                t_l_c_d = t_l_c_d + len(cc);

            ret.append("zstd_dict_" + str(i) + " dict_size: " + str(len(dict_data)) + " size: " + str(t_l_c_d) + " total: " + str(t_l_c_d+len(dict_data)));
        except:
            ret.append("zstd_"+str(i)+"_err");

    return ret;


def run(id, files):
    output_f = open("output/"+str(id), 'w');
    #output_err = open("output/"+str(id)+"_err", 'w');
    #sys.stderr = output_err;
    separator = '\t';
    files.sort(key=lambda f: os.stat(f).st_size, reverse=True);
    for file in files:
        if (file[-4:] == ".dat"):
            continue
        line = [file];

        f = open(file, "r+b");
        dmap = mmap.mmap(f.fileno(), 0);
        orig_size = os.path.getsize(file);
        line.append("orig_size: " + str(orig_size));

        b = bytes(dmap);
        sys.stderr.write(file + " START ******************************************************\n");
        #line = line + zlib_compress(b);
        #line = line + zstd_compress(b);
        line = line + zlib_strided_compress(b);
        #img = cv2.imread(file);
        #print(img.shape)
        f.close();
        sys.stderr.write(file + " END ******************************************************\n");


        #line.append(jpg_size);
        #pixel_size = img.shape[0]*img.shape[1]*img.shape[2];
        #line.append(pixel_size);
        #line = line + method1(img);
        #print("here1")
        #line = line + method2(img);
        #print("here2")
        #line = line + method3(img);
        #print("here3")
        #line = line + method4(img);
        #print("here4")
        #line = line + method5(img);
        #print("here5")

        out_string = separator.join(map(str, line)) + '\n';
        output_f.write(out_string);
        output_f.flush();
        #break;
    output_f.close();
    #output_err.close();

if __name__ == '__main__':
    files = [(PATH+f) for f in os.listdir(PATH) if os.path.isfile(os.path.join(PATH, f))];
    n_files = len(files);
    n_iters = n_files/N_PROCS;
    n_rem = n_files%N_PROCS;

    parent_conns = [None] * N_PROCS;
    child_conns = [None] * N_PROCS;
    processes = [None] * N_PROCS;

    for i in range(N_PROCS):
        begin = int(i*n_iters);
        end = int((i+1)*n_iters);
        processes[i] = multiprocessing.Process(target=run, args=(i, files[begin:end]));
        processes[i].start();
    for i in range(N_PROCS):
        processes[i].join();
