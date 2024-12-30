import pandas as pd

# 读取两个表格
cm_slava77_PR_comment_info = pd.read_excel('cm_slava77_PR_comment_info.xlsx')  # 根据实际文件路径调整
cm_PR_commit_info = pd.read_excel('cm_PR_commit_info.xlsx')  # 根据实际文件路径调整

# 根据 'commit_id' 列和 'sha' 列筛选数据
merged_data = pd.merge(cm_slava77_PR_comment_info, cm_PR_commit_info, left_on='commit_id', right_on='sha', how='inner')

# 保存筛选后的结果到新的 Excel 文件
merged_data.to_excel('filtered_cm_PR.xlsx', index=False)

print("筛选后的数据已保存到 filtered_cm_PR.xlsx")
