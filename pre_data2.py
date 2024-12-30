import pandas as pd

# 读取Excel文件
file_path = "D:\桌面\原始数据\processed_data.xlsx"  # 替换为你的文件路径
df = pd.read_excel(file_path)

# 筛选出diff_hunk列字数不少于3的行
df = df[df['diff_hunk'].str.len() >= 3]

# 删除diff_hunk列的重复行
df = df.drop_duplicates(subset=['diff_hunk'])

# 保存处理后的数据到新文件
output_path = "filtered_excel_file.xlsx"  # 替换为你希望保存的文件路径
df.to_excel(output_path, index=False)

print(f"处理后的文件已保存到: {output_path}")
