from dataset_handler import handler

# 2. صدا زدن تابع
X, y, raw_df, _ = handler(
    selected_dataset='iris',
    sample_name_column_name='',             # اگر ستونی مثل ID نداری، خالی بذار
    dropna=True,                            # حذف مقادیر گمشده
    irrelevant_feature_names=[],            # حذف ستون‌های بی‌ربط، اگر داری بنویس
    categorical_feature_names=[],           # ستون‌های دسته‌ای برای OneHot (مثلاً ['gender'] اگه داشتی)
    frac_of_samples_to_keep=1,              # 1 یعنی همه داده‌ها رو نگه داره
    seed_number=42,                         # عدد تصادفی برای ثبات نمونه‌گیری
    drop_duplicates=True,                   # حذف ردیف‌های تکراری
    return_X_y=True,                        # خروجی رو به صورت X و y بده
    output_folder_path='',                  # نیازی به ذخیره نیست فعلاً
    one_hot_encoding=False                  # چون ستون‌ها همه عددی‌ان
)

# 3. چاپ چند ردیف اول
print(X[:5])
print(y[:5])