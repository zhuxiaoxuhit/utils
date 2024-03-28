import json
import os
import string
import sys
from typing import List, Dict
import multiprocessing as mp
from multiprocessing import Process
import re
from itn.chinese.inverse_normalizer import InverseNormalizer
import glob


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
                        #cost = 0.5
                        cost = 0
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

def ProcessJson(jsonfile, asrmap: dict, outdir, itn):
    f = open(jsonfile, "r")
    lines = [line.strip() for line in f.readlines()]
    f.close()

    c2p = LoadC2P()

    objs : List[Dict] = []
    for line in lines:
        objs.append(json.loads(line))
    #

    for i, obj in enumerate(objs):
        if not obj["available_for_training"]:
            continue
        #
        wav_path = obj["wav_path"]
        text_ori = obj["text"]
        asrtext_ori = asrmap[wav_path]
        obj["asr_text"] = asrtext_ori
        text, idxtext = Levenshtein.text_norm(text_ori)
        asrtext, idxasr = Levenshtein.text_norm(asrtext_ori)
        alignresult = Levenshtein.min_edit_distance_with_alignment(c2p, text, asrtext, True)
        if alignresult[0] > len(text) / 2:
            obj["available_for_training"] = False
        obj["levenshtein_dist"] = alignresult[0]
        obj["levenshtein_rate"] = alignresult[0] / len(text) if len(text) > 0 else 1.0

        obj["levenshtein_di_dist"] = alignresult[3].count("D") + alignresult[3].count("I")
        obj["levenshtein_di_rate"] =  obj["levenshtein_di_dist"] / len(text) if len(text) > 0 else 1.0
        obj["levenshtein_di_inner_rate"] = obj["levenshtein_di_dist"] / obj["levenshtein_dist"] if obj["levenshtein_dist"] > 0 else 0

        obj["levenshtein_op"] = " ".join(alignresult[3])
        try:
            mergedtext_debug, mergedtext, mergeditn = MergeText(itn, c2p, text_ori, asrtext_ori, text, asrtext, idxtext, idxasr, alignresult[3])
            obj["merged_text"] = mergedtext
            obj["merged_text_debug"] = mergedtext_debug
            obj["merged_text_itn"] = mergeditn
        except Exception as e:
            sys.stderr.write(f"line {i} @ file {jsonfile} merge text failed\n")
            sys.stderr.flush()
            raise e
        #
    #
    
    f = open(outdir + "/" + os.path.basename(jsonfile), "w")
    for obj in objs:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")
    #
    f.close()
    
    f = open(outdir + "/" + os.path.basename(jsonfile) + ".endtag", "w")
    f.close()
#


def ProcessJsonList(jsonlist, asrmap, outdir):
    num = len(jsonlist)
    itn = InverseNormalizer()
    for i, f in enumerate(jsonlist):
        try:
            sys.stdout.write("processing: %s progress: %d / %d ==> %.4f\n" % (f, i + 1, num, (i + 1) / num))
            sys.stdout.flush()
            ProcessJson(f, asrmap, outdir, itn)
        except Exception as e:
            sys.stderr.write("\t FATAL: %s, %s\n" % (type(e).__name__, e.__str__()))
            sys.stderr.write(f"\t FATAL: error in processing: {f} @ stage levenshtein merge\n")
            sys.stderr.flush()
    #
#

def ProcessAll2(jsonlistfile, asrfile, outdir, ntask):
    asrmap = LoadAsr(asrfile)

    f = open(jsonlistfile)
    jsonlist = [line.strip() for line in f.readlines()]
    f.close()

    if ntask == 1:
        ProcessJsonList(jsonlist, asrmap, outdir)
    else:
        chunk = len(jsonlist) // ntask + 1
        tasklist = []
        for i in range(ntask):
            if i * chunk >= len(jsonlist):
                break
            tasklist.append(jsonlist[i * chunk : (i + 1) * chunk])
        ntask = len(tasklist)
        sys.stdout.write(f"total {ntask} actual parrallel tasks\n")
        sys.stdout.flush()

        subtasks = [mp.Process(target=ProcessJsonList, args=(tasklist[i], asrmap, outdir,)) for i in range(ntask)]

        [p.start() for p in subtasks]  
        [p.join() for p in subtasks]
#

def ProcessAll(jsonlistfile, asrfile, outdir, ntask):
    asrmap = LoadAsr(asrfile)

    f = open(jsonlistfile)
    jsonlist = [line.strip() for line in f.readlines()]
    f.close()

    pool = mp.Pool(ntask)
    tasks = mp.Queue()
    for f in jsonlist:
        tasks.put(f)
    #

    while not tasks.empty():
        print(tasks.qsize())
        task = tasks.get()
        pool.apply_async(ProcessJson, args=(task, asrmap, outdir, ))
    #
    pool.close()
    pool.join()
    #
#


if __name__ == "__main__":
    c2p = LoadC2P()
    a = Levenshtein.text_norm("最多饭软糯些水若少了不免成了夹生饭——这玩意儿只有评书里吞十斤烙饼有不锈钢肠胃的好汉爱吃但那时也不懂")
    b = Levenshtein.text_norm("最多饭软糯些水弱少了不免成了家生饭这玩意儿只有评书里吞十斤烙饼有不锈钢肠胃的好汉爱吃但那时也不")
    a = "最多饭软糯些水若少了不免成了夹生饭——这玩意儿只有评书里吞十斤烙饼有不锈钢肠胃的好汉爱吃但那时也不懂"
    b = "最多饭软糯些水弱少了不免成了家生饭这玩意儿只有评书里吞十斤烙饼有不锈钢肠胃的好汉爱吃但那时也不"
    itn = InverseNormalizer()
    print(itn.normalize(a))
    print(itn.normalize(b))
    out1, out2, out3, out4 = Levenshtein.min_edit_distance_with_alignment(c2p, itn.normalize(a), itn.normalize(b), True)
    print(out1)
    print(out2)
    print(out3)
    print(out4)
    #Levenshtein.display(out1, out2, out3, out4)
    #print(Levenshtein.text_norm("今天，天气，123接口。"))
    #TestMergeText()
    #c2p = LoadC2P()
    #print(SamePronounce(c2p, "征", "症"))
    #TestMergeText()
    #print(PostProcessMergedText("今天天气( ), 你好啊( ), 我很好（   ...）, “,,,”"))
    #print(PostProcessMergedText("“，。。今天天气( ), 你好啊( ), 我很好（  ,,,）, “,,,”"))

    #if len(sys.argv) < 5:
    #    sys.stderr.write(".py json-list, corresponding-asr-result, out-json-dir, ntask\n")
    #    exit(-1)
    ##
    #
    #ProcessAll2(sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]))
    #Process5k(c2p, sys.argv[1], sys.argv[2])

#
