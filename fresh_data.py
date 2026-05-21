import csv
from collections import Counter

# 读取原始数据文件
in_csv = "/home/fpk/project/IVD/Abdesign/flow_generated_CDR_sequences.csv"
out_csv = "clean_valid_CDR.csv"

valid_seqs = []
len_list = []

with open(in_csv, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        seq_len_str = row["序列长度"].strip()
        cdr_seq = row["CDR序列"].strip()
        try:
            seq_len = int(seq_len_str)
        except:
            continue
        
        # 筛选条件：5~30
        if 5 <= seq_len <= 30 and cdr_seq:
            valid_seqs.append(row)
            len_list.append(seq_len)

# 写入清洗后数据
with open(out_csv, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["抗体编号","序列长度","CDR序列"])
    writer.writeheader()
    writer.writerows(valid_seqs)

# 统计信息
total_all = len(list(csv.DictReader(open(in_csv, "r", encoding="utf-8"))))
total_valid = len(valid_seqs)
len_counter = Counter(len_list)

print("="*50)
print(f"原始总生成条数：{total_all}")
print(f"筛选后有效CDR条数：{total_valid}")
print(f"无效剔除条数：{total_all - total_valid}")
print("="*50)
print("序列长度分布(长度:数量)：")
for l in sorted(len_counter.keys()):
    print(f"{l:2d} 位 : {len_counter[l]} 条")
print("="*50)
print(f"最短有效长度：{min(len_list)}")
print(f"最长有效长度：{max(len_list)}")
print(f"✅ 干净序列已保存至：{out_csv}")