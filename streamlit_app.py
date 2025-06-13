import streamlit as st
import pandas as pd
from PIL import Image
import torch
import os
import warnings
warnings.filterwarnings('ignore', category=UserWarning)

# 设置页面标题
st.title("🐱 识别与推荐系统")

# 设置自定义字体
st.markdown(
    """
    <style>
    * {
        font-family: "华文彩云", sans-serif !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# 读取数据
@st.cache_data
def load_data():
    try:
        nature_df = pd.read_excel('cat性格矩阵.xlsx')
        cats_df = pd.read_excel('cats.xlsx')
        merged_df = pd.merge(nature_df, cats_df, on='cat_id', how='left', suffixes=('_nature', '_cats'))
        merged_df['cat'] = merged_df['cat_cats']
        merged_df.drop(['cat_nature', 'cat_cats'], axis=1, inplace=True, errors='ignore')
        
        # 处理性格矩阵
        all_natures = ['粘人', '独立', '好动', '安静', '好奇心强', '好奇心弱', '易训练', '难训练', 
                      '梳理需求高', '梳理需求低', '长毛', '短毛', '无毛', '亲人程度高', '亲人程度低']
        for nature in all_natures:
            merged_df[nature] = merged_df[nature].apply(lambda x: 1 if pd.notna(x) and x != 0 else 0)
        
        return merged_df
    except Exception as e:
        st.error(f"加载数据时出错: {str(e)}")
        return pd.DataFrame()

# 推荐函数
def recommend_cats(user_prefs, merged_df):
    if merged_df.empty:
        return []
    
    merged_df['match_score'] = merged_df.apply(
        lambda row: sum(row[pref] * user_prefs[pref] for pref in user_prefs),
        axis=1
    )
    
    recommended = merged_df.sort_values(by='match_score', ascending=False).head(3)
    results = []
    
    for _, row in recommended.iterrows():
        cat_name = row['cat']
        image_ext = next((ext for ext in ['.jpg', '.png', '.webp'] 
                         if os.path.exists(f'猫/{cat_name}{ext}')), None)
        
        results.append({
            '品种': cat_name,
            '介绍': row['cats'],
            '图片路径': f'猫/{cat_name}{image_ext}' if image_ext else None
        })
    
    return results

# 加载模型函数
def load_cat_model():
    try:
        # 仅在需要时导入fastai
        from fastai.vision.all import load_learner
        return load_learner('12cat_model.pkl')
    except Exception as e:
        st.error(f"加载模型时出错: {str(e)}")
        return None

# 主程序
merged_df = load_data()

# 创建标签页
tab1, tab2 = st.tabs(["🐾 猫品种识别", "❤️ 个性化推荐"])

with tab1:
    # 文件上传组件
    uploaded_file = st.file_uploader(
        "请上传猫的图片", 
        type=["jpg", "png", "webp", "jpeg"],
        help="支持JPG/PNG/WEBP格式"
    )

    if uploaded_file is not None:
        # 显示上传的图片
        img = Image.open(uploaded_file)
        st.image(img, caption="您上传的图片", use_container_width=True)
        
        try:
            # 仅在需要时加载模型
            learn = load_cat_model()
            
            if learn:
                # 进行预测
                pred, _, probs = learn.predict(img)
                
                # 显示结果
                st.success(f'识别结果: {pred}')
                
                # 显示概率(可选)
                with st.expander("查看详细概率"):
                    st.write("各品种概率:")
                    for i, (cat, prob) in enumerate(zip(learn.dls.vocab, probs)):
                        st.write(f"{cat}: {prob*100:.2f}%")
            else:
                st.error("无法加载模型，请检查模型文件是否存在")
        except Exception as e:
            st.error(f"识别时出错: {str(e)}")
            st.info("请确保模型文件存在且格式正确")

with tab2:
    st.header("根据您的偏好推荐猫品种")
    
    # 用户偏好选择
    st.subheader("请选择您的偏好（每个类别选一个）：")
    
    col1, col2 = st.columns(2)
    
    with col1:
        personality = st.radio("性格", ["粘人", "独立"])
        activity = st.radio("活动量", ["好动", "安静"])
        curiosity = st.radio("好奇心", ["好奇心强", "好奇心弱"])
        trainability = st.radio("可训练性", ["易训练", "难训练"])
    
    with col2:
        grooming = st.radio("梳理需求", ["梳理需求高", "梳理需求低"])
        fur_type = st.radio("毛发类型", ["长毛", "短毛", "无毛"])
        affection = st.radio("亲人程度", ["亲人程度高", "亲人程度低"])
    
    # 设置用户偏好
    user_prefs = {
        '粘人': 0, '独立': 0,
        '好动': 0, '安静': 0,
        '好奇心强': 0, '好奇心弱': 0,
        '易训练': 0, '难训练': 0,
        '梳理需求高': 0, '梳理需求低': 0,
        '长毛': 0, '短毛': 0, '无毛': 0,
        '亲人程度高': 0, '亲人程度低': 0
    }
    
    user_prefs[personality] = 1
    user_prefs[activity] = 1
    user_prefs[curiosity] = 1
    user_prefs[trainability] = 1
    user_prefs[grooming] = 1
    user_prefs[fur_type] = 1
    user_prefs[affection] = 1
    
    if st.button("获取推荐"):
        recommendations = recommend_cats(user_prefs, merged_df)
        
        st.subheader("为您推荐的猫品种：")
        
        for rec in recommendations:
            # 创建左右分栏布局 (1:5比例)
            col1, col2 = st.columns([1, 5])
            
            with col1:
                # 显示猫咪图片，占1/6宽度
                if rec['图片路径']:
                    st.image(rec['图片路径'], use_container_width=True)
                else:
                    st.warning("未找到该品种的图片")
            
            with col2:
                # 显示品种名称和介绍
                st.markdown(f"### {rec['品种']}")
                st.write(rec['介绍'])
            
            st.write("---")
