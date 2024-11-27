from bs4 import BeautifulSoup
# from googletrans import Translator
import re
import time
import pandas as pd
from deep_translator import GoogleTranslator

#--------------------------------------------------------------------------------------------------

def get_cbcl_details(cbcl_item):
    """
    根据提供的 cbcl_q 字段组合（如 "avg_cbcl_q08_p_cbcl_q10_p_cbcl_q78_p"）从 element.html 文件中获取详细信息。
    
    参数:
        cbcl_item (str): 要查找的 cbcl_q 字段组合（如 "avg_cbcl_q08_p_cbcl_q10_p_cbcl_q78_p"）。
    
    返回:
        str: 详细信息的组合，如果找不到则返回 "N/A"。
    """
    # 解析 element.html 文件
    with open("data/element.html", "r", encoding="utf-8") as file:
        soup = BeautifulSoup(file, "html.parser")

    # 使用正则表达式提取所有的 cbcl_q 字段
    cbcl_pattern = re.compile(r"(cbcl_q\d+[a-z]*_p)")
    cbcl_items = cbcl_pattern.findall(cbcl_item)
    
    # 存储每个 cbcl 字段的详细信息
    details = []

    for cbcl in cbcl_items:
        # 在 HTML 中查找包含 cbcl 的 <td> 标签
        target = soup.find(lambda tag: tag.name == "td" and cbcl in tag.get_text(strip=True))
        
        # 获取详细信息
        if target:
            detail_info = target.find_next("td").get_text(strip=True)
            details.append(detail_info)
        else:
            details.append("N/A")
    
    # 合并所有详细信息为一个字符串
    combined_details = "; ".join(details) if details else "N/A"

    return combined_details

# 示例调用
detail = get_cbcl_details("avg_cbcl_q08_p_cbcl_q10_p_cbcl_q78_p")
print("详细信息:", detail)



#--------------------------------------------------------------------------------------------------
""" 输入数据框DF(形如Row_Name1,Row_Name2),因子数量和语言(string), 返回一个包含翻译后详细信息的数据框"""

def translate_text(df, number_of_factors,language):

    # 解析 element.html 文件以获取列名和详细信息
    with open("data/element.html", "r", encoding="utf-8") as file:
        soup = BeautifulSoup(file, "html.parser")

    # 创建一个字典来存储列名和对应的详细信息
    column_details = {}
    result_df = pd.DataFrame()

    # 提取 cbcl_q 列名的正则表达式
    cbcl_pattern = re.compile(r"(cbcl_q\d+[a-z]*_p)")

    for i in range(0, number_of_factors):
        # 筛选出符合条件的加载值
        # factor_values = df[f"Factor {i}"][df[f"Factor {i}"] > 0.1]
        
        original_text = []
        translated_text = []
        for column_name in df.iloc[:,i]:
            # 查找 column_name 中的所有 cbcl_q 字段
            cbcl_items = cbcl_pattern.findall(column_name)  # 提取所有符合 cbcl_qXX_p 或 cbcl_qXXh_p 格式的子串

            # 初始化存储每个 cbcl 字段详细信息的列表
            original = []
            details = []
            for cbcl_item in cbcl_items:
                # 获取每个 cbcl 字段的详细信息
                target = soup.find(lambda tag: tag.name == "td" and cbcl_item in tag.get_text(strip=True))
                if target:
                    detail_info = target.find_next("td").get_text(strip=True)
                    # 保存原始详细信息
                    original.append(detail_info)
                    
                    # 翻译详细信息并添加到结果
                    try:
                        translated_detail = GoogleTranslator(source='es', target=language).translate(detail_info)
                    except AttributeError as e:
                        print(f"An error occurred: {e}")
                        translated_detail = detail_info
                    details.append(translated_detail)
                    time.sleep(0.25)

            # 将所有细节合并为单个字符串，并添加到列表中
            original_text.append("; ".join(original) if original else "N/A")
            translated_text.append("; ".join(details) if details else "N/A")
        # 创建一个临时数据框保存因子名、列名、加载值和详细信息
        temp_df = pd.DataFrame({
            # f"Factor {i} Variable": factor_values.index,  # 存储列名
            # f"Factor {i} Loading": factor_values.values,  # 存储加载值
            f"Factor {i} Detail": original_text,  # 映射详细信息
            f"Factor {i} Translated_Detail": translated_text  # 映射翻译后详细信息
        })

        # 按加载值降序排序
        # sorted_df = temp_df.sort_values(by=f"Factor {i} Loading", ascending=False).reset_index(drop=True)
        # 将临时数据框合并到结果数据框
        result_df = pd.concat([result_df.reset_index(drop=True), temp_df.reset_index(drop=True)], axis=1)
    return result_df