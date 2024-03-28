import re
import time
import argparse 
import sys
import os
import math
import shutil
import glob
#import jieba
import numpy as np
import pandas as pd
#from simhash import Simhash, SimhashIndex
import chardet
#import codecs
import multiprocessing as mp
from multiprocessing import Process
import string
from typing import List, Dict
from itn.chinese.inverse_normalizer import InverseNormalizer


chinese_punctuation = "！？｡。·＇：∶、，；《》「」『』〖〗【】（）〔〕［］｛｝“”‘’〝〞、︵︷︿︹︽︿︺︶︸︼︺︾﹁﹃﹂﹄﹏﹋﹌—…～｜‖＂"


class Levenshtein():
    @staticmethod
    def text_norm(str1):
        # 匹配汉字、英文单词和数字的正则表达式
        pattern = re.compile(r'[\u4e00-\u9fa5]|[a-zA-Z0-9_.]+')

        # 使用正则表达式匹配字符串中的所有符合条件的子字符串
        matches = [match for match in pattern.finditer(str1)]
        matchstr = [match.group() for match in matches]
        matchidx = [match.start() for match in matches]

        return matchstr, matchidx

    @staticmethod
    def min_edit_distance_with_alignment(c2p, str1, str2, align=False):
        len_str1 = len(str1) + 1
        len_str2 = len(str2) + 1
    
        # 初始化一个二维数组，用于存储编辑距离的中间结果
        dp = [[0] * len_str2 for _ in range(len_str1)]
    
        # 初始化记录编辑操作的二维数组
        operations = [[""] * len_str2 for _ in range(len_str1)]
    
        # 初始化第一行和第一列
        for i in range(len_str1):
            dp[i][0] = i
            operations[i][0] = "D"  # Delete
        for j in range(len_str2):
            dp[0][j] = j
            operations[0][j] = "I"  # Insert
    
        # 动态规划计算编辑距离和记录编辑操作
        for i in range(1, len_str1):
            for j in range(1, len_str2):
                #cost = 0 if str1[i - 1] == str2[j - 1] else 1
                if str1[i - 1].lower() == str2[j - 1].lower():
                    cost = 0
                else:
                    if SamePronounce(c2p, str1[i - 1], str2[j - 1]):
                        cost = 0.5
                    else:
                        cost = 1

                # 计算编辑距离
                delete = dp[i - 1][j] + 1
                insert = dp[i][j - 1] + 1
                substitute = dp[i - 1][j - 1] + cost
    
                dp[i][j] = min(delete, insert, substitute)
    
                # 记录编辑操作
                if dp[i][j] == delete:
                    operations[i][j] += "D"
                if dp[i][j] == insert:
                    operations[i][j] += "I"
                if dp[i][j] == substitute:
                    operations[i][j] += "S" if cost > 0 else "M"  # S for substitute, M for match
        if not align:
            return dp[len_str1 - 1][len_str2 - 1], None, None, None

        # 回溯，构造对齐和编辑方式
        align1 = []
        align2 = []
        alignop = []
        align_str1 = ""
        align_str2 = ""
        i, j = len_str1 - 1, len_str2 - 1
    
        while i > 0 or j > 0:
            if "D" in operations[i][j]:
                align_str1 = str1[i - 1] + align_str1
                align_str2 = "--" + align_str2
                align1.insert(0, str1[i - 1])
                align2.insert(0, "-")
                alignop.insert(0, "D")
                i -= 1
            elif "I" in operations[i][j]:
                align_str1 = "--" + align_str1
                align_str2 = str2[j - 1] + align_str2
                align1.insert(0, "-")
                align2.insert(0, str2[j - 1])
                alignop.insert(0, "I")
                j -= 1
            elif "S" in operations[i][j]:
                align_str1 = str1[i - 1] + align_str1
                align_str2 = str2[j - 1] + align_str2
                align1.insert(0, str1[i - 1])
                align2.insert(0, str2[j - 1])
                alignop.insert(0, "S")
                i -= 1
                j -= 1
            elif "M" in operations[i][j]:
                align_str1 = str1[i - 1] + align_str1
                align_str2 = str2[j - 1] + align_str2
                align1.insert(0, str1[i - 1])
                align2.insert(0, str2[j - 1])
                alignop.insert(0, "_")
                i -= 1
                j -= 1
        return dp[len_str1 - 1][len_str2 - 1], align1, align2, alignop
    
    @staticmethod
    def display(dist, align1, align2, alignop):
        print(f"The minimum edit distance is: {dist}")
        for item in align1:
            print(f"{item}".ljust(8), end="")
        #
        print("")
        for item in align2:
            print(f"{item}".ljust(8), end="")
        #
        print("")
        for item in alignop:
            print(f"{item}".ljust(8), end="")
        #
        print("")
#

def LoadAsr(asrfile):
    f = open(asrfile, "r")
    lines = [line.strip() for line in f.readlines()]
    f.close()
    asrmap = dict()
    for line in lines:
        tmp = line.split("\t")
        if len(tmp) != 2:
            asrmap[tmp[0]] = ""
            sys.stderr.write(f"Warning: invalidate format @ line {line} @ asr result file {asrfile}\n")
            sys.stderr.flush()
            continue
        asrmap[tmp[0]] = tmp[1]
    #
    return asrmap

def LoadC2P():
    c2p : Dict[List[str]] = dict()
    f = open("/home/zhuxiaoxu/t3/monophonic_charcters.txt")
    lines = [line.strip() for line in f.readlines()]
    for line in lines:
        tmp = line.split()
        if tmp[0] not in c2p:
            c2p[tmp[0]] = []
        c2p[tmp[0]].append(tmp[1][:-1])
    f.close()

    f = open("/home/zhuxiaoxu/t3/polyphonic_charcters.txt")
    lines = [line.strip() for line in f.readlines()]
    for line in lines:
        tmp = line.split()
        if tmp[0] not in c2p:
            c2p[tmp[0]] = []

        for i in tmp[1:]:
            if "(" not in i:
                c2p[tmp[0]].append(i[:-1])
            else:
                patten = "[^a-z]"
                x = re.sub(patten, "", i)
                c2p[tmp[0]].append(x)
    f.close()

    f = open("/home/zhuxiaoxu/t3/polyphonic_rule_charcters.txt")
    lines = [line.strip() for line in f.readlines()]
    for line in lines:
        tmp = line.split()
        if tmp[0] not in c2p:
            c2p[tmp[0]] = []

        for i in tmp[1:]:
            if "(" not in i:
                c2p[tmp[0]].append(i[:-1])
            else:
                patten = "[^a-z]"
                x = re.sub(patten, "", i)
                c2p[tmp[0]].append(x)
    f.close()
    
    return c2p

def SamePronounce(c2p, a, b) -> bool:
    if a == b:
        return True

    if a not in c2p or b not in c2p:
        return False
    
    if len(set(c2p[a]) & set(c2p[b])) > 0:
        return True

    return False
#

def WithPunc(text, text_norm, idx, i):
    if i < len(idx) - 1:
        return idx[i + 1] - idx[i] > len(text_norm[i])
    else:
        return len(text) - idx[i] > len(text_norm[i])
#

def PostProcessMergedText(text):
    text = re.sub("\([^0-9a-zA-Z\u4e00-\u9fa5]*\)", "", text)
    text = re.sub("（[^0-9a-zA-Z\u4e00-\u9fa5]*）", "", text)
    text = re.sub("“[^0-9a-zA-Z\u4e00-\u9fa5]*”", "", text)
    text = re.sub("“[,，。?？!！;；、~]+", "“", text)
    return text

def MergeText(itn, c2p, text: str, asr, norm_text, norm_asr, idx_text, idx_asr, op_list):
    merge = []
    mergeFinal = []
    ct = 0
    ca = 0
    if len(norm_text) > 0 and idx_text[0] > 0:
        merge.append(text[0:idx_text[0]])
        mergeFinal.append(text[0:idx_text[0]])
    for i, op in enumerate(op_list):
        if op == "_":
            ts = idx_text[ct]
            te = idx_text[ct + 1] if ct < len(idx_text) - 1 else len(text)
            merge.append(text[ts:te])
            mergeFinal.append(text[ts:te])
            ct += 1
            ca += 1
        elif op == "S":
            ts = idx_text[ct]
            te = idx_text[ct + 1] if ct < len(idx_text) - 1 else len(text)
            cont = text[ts:te]
            if not SamePronounce(c2p, norm_text[ct], norm_asr[ca]):
                contD = cont.replace(norm_text[ct], "*" + norm_asr[ca] + "*")
                contF = cont.replace(norm_text[ct], norm_asr[ca])
            else:
                contD = cont.replace(norm_text[ct], "^" + norm_text[ct] + "^")
                contF = cont.replace(norm_text[ct], norm_text[ct])
                
            merge.append(contD)
            mergeFinal.append(contF)
            ct += 1
            ca += 1
        elif op == "I":
            if ct > 0:
                s = idx_text[ct - 1]
                if ct < len(idx_text):
                    e = idx_text[ct]
                else:
                    e = len(text)
                match = re.search(r"[^a-zA-Z0-9\u4e00-\u9fa5\*\(\)\{\}]", merge[-1])
                matchF = re.search(r"[^a-zA-Z0-9\u4e00-\u9fa5]", mergeFinal[-1])
                if (e - s) != len(norm_text[ct - 1]) and match:
                    # last char ends with punc, 还得判断不是:""
                    if (norm_asr[ca] in "儿呀啊呐哎吗呢的了" or WithPunc(asr, norm_asr, idx_asr, ca)) and \
                        "“" not in merge[-1][match.start():]:
                        ### To Do, debug symbol
                        merge[-1] = merge[-1][:match.start()] + "(" + norm_asr[ca] + ")" + merge[-1][match.start():]
                        mergeFinal[-1] = mergeFinal[-1][:matchF.start()] + norm_asr[ca] + mergeFinal[-1][matchF.start():]
                    else:
                        merge.append("(" + norm_asr[ca] + ")")
                        mergeFinal.append(norm_asr[ca])
                else:
                    merge.append("{" + norm_asr[ca] + "}")
                    mergeFinal.append(norm_asr[ca])
            else:
                merge.append(norm_asr[ca])
                mergeFinal.append(norm_asr[ca])
            ca += 1
        elif op == "D":
            if ct == len(idx_text) - 1 and i > 0 and op_list[i - 1] != 'D':
            # if D is last char, keep it, since asr may lost last char
                ts = idx_text[ct]
                te = idx_text[ct + 1] if ct < len(idx_text) - 1 else len(text)
                merge.append(text[ts:te])
                mergeFinal.append(text[ts:te])
                ct += 1
            else:
                ts = idx_text[ct]
                te = idx_text[ct + 1] if ct < len(idx_text) - 1 else len(text)
                cont = text[ts:te]
                cont = cont.replace(norm_text[ct], "")
                merge.append(cont)
                mergeFinal.append(cont)
                ct += 1
        else:
            raise Exception("")
    if ct != len(norm_text) or ca != len(norm_asr):
        raise Exception("")
    
    posttext = PostProcessMergedText("".join(mergeFinal))
    itntext = itn.normalize(posttext)
                                
    return "".join(merge), posttext, itntext
#


def extract_all_quotes_from_files(file_list):
    """
    This function takes a list of .txt file paths, reads each file, extracts content enclosed
    in both English and Chinese double quotes, and returns a list of all such quoted content found across the files.
    
    Args:
    file_list (list of str): The list of file paths to .txt files to be processed.
    
    Returns:
    list of str: A list containing all pieces of text found within both English and Chinese double quotes across all files.
    """
    # List to hold all the quoted texts found
    quoted_contents = []
    
    # Iterate over each file in the provided list
    for file_path in file_list:
        # Open and read the content of each file
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
                # Find all instances of text enclosed in English and Chinese double quotes and add them to the list
                # We use a regular expression to find text within quotes
                quoted_contents.extend(re.findall(r'["“](.*?)["”]', content))
    
    # Return the list of all quoted contents found
    return quoted_contents

# 文本相似度1：计算交集大小/并集大小
def calculate_similarity_set(asr_text, origin_text):
    # 去标点
    normalized_asr_text =  re.sub(r'[^\w\s]+', '', asr_text.lower())
    normalized_origin_text = re.sub(r'[^\w\s]+', '', origin_text.lower())
    intersection = len(set(normalized_asr_text) & set(normalized_origin_text))
    union = len(set(normalized_asr_text)) + len(set(normalized_origin_text))
    if union == 0:
        union = 100000 
    return intersection / union

# 计算编辑距离 
def calculate_edit_distance(str1_o, str2_o):
    # 去标点
    str1 = re.sub(r'[^\w\s]+', '', str1_o.lower())
    str2 = re.sub(r'[^\w\s]+', '', str2_o.lower())

    m, n = len(str1), len(str2)
    # 创建一个矩阵来存储子问题的解
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    # 初始化矩阵的第一行和第一列
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    # 使用动态规划填充矩阵
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if str1[i - 1] == str2[j - 1]:
                cost = 0
            else:
                cost = 1
            dp[i][j] = min(dp[i - 1][j] + 1,      # 删除
                           dp[i][j - 1] + 1,      # 插入
                           dp[i - 1][j - 1] + cost)  # 替换

    # 返回两个字符串的编辑距离
    return dp[m][n]


def process(args, cp2, itn):
    ### 读取文本音频 .xlsx 文件
    df = pd.read_excel(io=args.input_xls_file)
    #fout = open(args.output_scripts_dir, "w")
    #fout_good = open(args.output_scripts_dir_good, "w")
    for index, row in df.iterrows():
        if str(row["音频"]) != 'nan' and str(row["书链接"]) != 'nan' and str(row["文本"]) != 'nan':
            str_cur_book_links = str(row["书链接"])
            audio_path = os.path.basename(str(row["音频"]))
            asr_text = str(row["文本"]).strip()
            asr_text_itn = itn.normalize(str(row["文本"]).strip())
            book_list = str_cur_book_links.strip().strip(",").split(",")
            book_list = [os.path.join(args.input_book_dir, os.path.basename(x).split(".")[0] + ".txt") for x in book_list]
            original_text_quotes_lst = extract_all_quotes_from_files(book_list)
            if len(original_text_quotes_lst) == 0:
                continue
            
            
            min_lenvenstin_value = 1000
            best_text = original_text_quotes_lst[0]
            for origin_text in original_text_quotes_lst:
                
                origin_text_itn = itn.normalize(origin_text)
                #dis = calculate_edit_distance(asr_text, origin_text)
                dis, _, _, detail  = Levenshtein.min_edit_distance_with_alignment(c2p, asr_text_itn, origin_text_itn, True)
                levenstin_rate = dis / len(asr_text_itn)
                #print(dis)
                #if abs(asr_text_itn - origin_text_itn) <= 4: 
                if min_lenvenstin_value > levenstin_rate:
                    best_text = origin_text
                    min_lenvenstin_value = levenstin_rate
            #if min_lenvenstin_value < 1:
                best_text = itn.normalize(best_text)
            final_lenvenstin_value = round(min_lenvenstin_value,2)
            fout.write(audio_path + "\t" + best_text + "\t" + asr_text + "\t" + str(final_lenvenstin_value) + "\n")
            if final_lenvenstin_value < 0.8:
                fout_good.write(audio_path + "\t" + best_text + "\t" + asr_text + "\t" + str(final_lenvenstin_value) + "\n")
           
            
            #max_sim_value = 0
            #best_text = original_text_quotes_lst[0]
            #for origin_text in original_text_quotes_lst:
            #    sim = calculate_similarity_set(asr_text, origin_text)
            #    if max_sim_value > sim:
            #        best_text = origin_text
            #        max_sim_value = sim
            #fout.write(audio_path + "\t" + best_text + "\t" + asr_text + "\n") 

    #with Pool(processes=multiprocessing.cpu_count()-3) as p:
    #|   p.map(download_audio_and_text_sub, download_list)

    #c = 0
    #for i in download_list:
    #    print(i)
    #    if c<= 100:
    #        download_audio_and_text_sub(i)
    #        c += 1
    #    else:
    #        break

#def set_dirs():
#    not os.path.exists(args.output_book_dir) and os.makedirs(args.output_book_dir)



def process_one(args, cp2, itn, row):
    try:
        if str(row["音频"]) != 'nan' and str(row["书链接"]) != 'nan' and str(row["文本"]) != 'nan':
            str_cur_book_links = str(row["书链接"])
            audio_path = os.path.basename(str(row["音频"]))
            asr_text = str(row["文本"]).strip()
            asr_text_itn = itn.normalize(str(row["文本"]).strip())
            book_list = str_cur_book_links.strip().strip(",").split(",")
            book_list = [os.path.join(args.input_book_dir, os.path.basename(x).split(".")[0] + ".txt") for x in book_list]
            original_text_quotes_lst = extract_all_quotes_from_files(book_list)
            if len(original_text_quotes_lst) == 0:
                return
            
            
            min_lenvenstin_value = 1000
            best_text = original_text_quotes_lst[0]
            for origin_text in original_text_quotes_lst:
                
                origin_text_itn = itn.normalize(origin_text)
                #dis = calculate_edit_distance(asr_text, origin_text)
                dis, _, _, detail  = Levenshtein.min_edit_distance_with_alignment(c2p, asr_text_itn, origin_text_itn, True)
                levenstin_rate = dis / len(asr_text_itn)
                #print(dis)
                if len(asr_text_itn) == 0 or len(origin_text_itn) == 0:
                    continue
                if (len(asr_text_itn) / len(origin_text_itn) > 0.8 and len(asr_text_itn) / len(origin_text_itn) < 1.2) or abs(len(asr_text_itn) - len(origin_text_itn)) < 2: 
                    if min_lenvenstin_value > levenstin_rate:
                        best_text = origin_text
                        min_lenvenstin_value = levenstin_rate
            best_text = itn.normalize(best_text)
            final_lenvenstin_value = round(min_lenvenstin_value, 2)
            with open(os.path.join(args.output_scripts_dir, audio_path+".txt") , "w", encoding='utf-8') as fout:
                fout.write(audio_path + "\t" + best_text + "\t" + asr_text + "\t" + str(final_lenvenstin_value) + "\n")
            if final_lenvenstin_value < 0.8:
                with open(os.path.join(args.output_scripts_dir_good, audio_path+".txt") , "w", encoding='utf-8') as fout_good:
                    fout_good.write(audio_path + "\t" + best_text + "\t" + asr_text + "\t" + str(final_lenvenstin_value) + "\n")
    except:
        print("ERROR: " + str(row["文本"]))


def process_mp(args, cp2, itn):
    ### 读取文本音频 .xlsx 文件
    df = pd.read_excel(io=args.input_xls_file)
    num_cores = mp.cpu_count()
    pool = mp.Pool(num_cores)

    args_list = [(args, cp2, itn, row) for index, row in df.iterrows()]

    pool.starmap(process_one, args_list)

    pool.close()
    pool.join()


def set_dirs():
    not os.path.exists(args.output_scripts_dir) and os.makedirs(args.output_scripts_dir)
    not os.path.exists(args.output_scripts_dir_good) and os.makedirs(args.output_scripts_dir_good)


if __name__ == "__main__":
    """
    usage
    python wav_preprocess.py -i input_dir -o output_dir
    """
    start_time = time.time()  # 统计时间
    parser = argparse.ArgumentParser()
    parser.add_argument('-ix', '--input_xls_file', type=str)
    #parser.add_argument('-oa', '--output_audio_dir', type=str)
    parser.add_argument('-ib', '--input_book_dir', type=str)
    parser.add_argument('-os', '--output_scripts_dir', type=str)
    parser.add_argument('-osg', '--output_scripts_dir_good', type=str)
    args = parser.parse_args()
    c2p = LoadC2P()
    itn = InverseNormalizer()
    set_dirs()
    process_mp(args, c2p, itn)
    print('time elapsed: ', time.time() - start_time, 's')


#if __name__ == "__main__":
#    c2p = LoadC2P()
#    a = Levenshtein.text_norm("最多饭软糯些水若少了不免成了夹生饭——这玩意儿只有评书里吞十斤烙饼有不锈钢肠胃的好汉爱吃但那时也不懂")
#    b = Levenshtein.text_norm("最多饭软糯些水弱少了不免成了家生饭这玩意儿只有评书里吞十斤烙饼有不锈钢肠胃的好汉爱吃但那时也不")
#    a = "最多饭软糯些水若少了不免成了夹生饭——这玩意儿只有评书里吞十斤烙饼有不锈钢肠胃的好汉爱吃但那时也不懂"
#    b = "最多饭软糯些水弱少了不免成了家生饭这玩意儿只有评书里吞十斤烙饼有不锈钢肠胃的好汉爱吃但那时也不"
#    itn = InverseNormalizer()
#    print(itn.normalize(a))
#    print(itn.normalize(b))
#    out1, out2, out3, out4 = Levenshtein.min_edit_distance_with_alignment(c2p, itn.normalize(a), itn.normalize(b), True)
#    print(out1)
#    print(out2)
#    print(out3)
#    print(out4)
#    #Levenshtein.display(out1, out2, out3, out4)
#    #print(Levenshtein.text_norm("今天，天气，123接口。"))
#    #TestMergeText()
#    #c2p = LoadC2P()
#    #print(SamePronounce(c2p, "征", "症"))
#    #TestMergeText()
#    #print(PostProcessMergedText("今天天气( ), 你好啊( ), 我很好（   ...）, “,,,”"))
#    #print(PostProcessMergedText("“，。。今天天气( ), 你好啊( ), 我很好（  ,,,）, “,,,”"))
#
#    #if len(sys.argv) < 5:
#    #    sys.stderr.write(".py json-list, corresponding-asr-result, out-json-dir, ntask\n")
#    #    exit(-1)
#    ##
#    #
#    #ProcessAll2(sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]))
#    #Process5k(c2p, sys.argv[1], sys.argv[2])
#
##
