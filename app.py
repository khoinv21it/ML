import tkinter as tk
from tkinter import Label, ttk, filedialog, messagebox
from turtle import color, left
from matplotlib import figure
from matplotlib.colors import ListedColormap
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn.ensemble import RandomForestClassifier
from ttkthemes import ThemedStyle
import seaborn as sns

from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVC
from sklearn.cluster import KMeans

# Recommendation
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Confussion matrix
from sklearn.metrics import classification_report, confusion_matrix
import itertools
import numpy as np

from scipy import stats

# INFORMATION

df = pd.DataFrame
model_predict = None

def load_csv_variables():
    global variables
    global df
    file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
    if file_path:
        try:
            with open(file_path, 'r') as file:
                global df 
                df = pd.read_csv(file_path)               
                variables = list(df.columns)
                input_variable_listbox.delete(0, tk.END)
                input_variable_listbox.insert(tk.END, *variables)
                product_description_listbox.delete(0, tk.END)
                product_description_listbox.insert(tk.END, *variables)
                target_variable_combobox['values'] = list(df.columns)
                select_variable_combobox['values'] = list(df.columns)
                display_variable_combobox['values'] = list(df.columns)
                selected_input_variables_listbox.delete(0, tk.END)
                target_variable_combobox.set('')
                select_variable_combobox.set('')
                load_dataset()
                missingvalue()
                plot_correlation_matrix()
                scatter(df.columns)
        except Exception as e:
            status_label.config(text=f"Error loading CSV file: {e}", foreground="red")

#Load biến vào ô input
def load_variable_info(selected_variable):
    variable_info_label = tk.Label(tab1, text=f"Information for variable: {selected_variable}", font=("Arial", 12, "bold"), bg="white")
    variable_info_label.grid(row=3, column=0, padx=10, pady=6, sticky="nsew")
    
    missing_values_label = tk.Label(tab1, text=f"Missing values: {df[selected_variable].isnull().sum()}", bg="white")
    missing_values_label.grid(row=4, column=0, padx=10, pady=2, sticky="w")
    
    unique_label = tk.Label(tab1, text="Unique values:", font=("Arial", 10, "bold"), bg="white")
    unique_label.grid(row=5, column=0, padx=10, pady=1, sticky="w")
  
    unique_text = tk.Text(tab1, wrap="word", height=6, width=40)
    unique_text.grid(row=6, column=0, padx=10, pady=2, sticky="nsew")
    unique = df[selected_variable].value_counts()
    unique_text.insert(tk.END, "\n".join(f"Value: {value}   -   count: {count}" for value, count in unique.items()))
    unique_text.configure(state="disabled")
    
    plot_variable(selected_variable)

def plot_variable(selected_variable):
    for widget in tab7.winfo_children():
        widget.destroy()

    plt.figure(figsize=(8, 6))
    # selected_variable = variable_combobox.get()

    if df[selected_variable].dtype != "object":
        sns.histplot(df[selected_variable], kde=True)
        plt.title(f'Distribution of {selected_variable}')
        plt.xlabel(selected_variable)
        plt.ylabel('Frequency')
    else:
        sns.countplot(x=selected_variable, data=df)
        plt.title(f'Count Plot of {selected_variable}')
        plt.xlabel(selected_variable)
        plt.ylabel('Count')

    # Tạo FigureCanvasTkAgg và hiển thị trên tab7
    canvas = FigureCanvasTkAgg(plt.gcf(), master=tab7)
    canvas.draw()
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

#Hiển thị thông tin dataset
def load_dataset():
    # Hiển thị thông tin của dataset ở panel2
    for widget in tab1.winfo_children():
        widget.destroy()
    info_label = tk.Label(tab1, text="Preview of Dataset", font=("Tahoma", 12, "bold"))
    info_label.grid(row=0, column=0, padx=10, pady=5)
                    
    # Hiển thị thông tin 5 dòng đầu và 5 dòng cuối
    preview_df = pd.concat([df.head(), df.tail()])
    tree = ttk.Treeview(tab1, columns=list(preview_df.columns), show="headings")
    for col in preview_df.columns:
        tree.heading(col, text=col)
        tree.column(col, width=80)
    for index, row in preview_df.iterrows():
        tree.insert("", "end", values=list(row))
    tree.grid(row=1, column=0, padx=10, pady=5, sticky="nsew")
                    
    # Hiển thị thông tin của dataset
    info_text = tk.Text(tab1, wrap="word", height=10, width=80)
    info_text.grid(row=2, column=0, padx=10, pady=5, sticky="nsew")
                    
    info_text.insert(tk.END, "Basic information about the dataset:\n\n")
    info_text.insert(tk.END, f"Number of rows: {len(df)}\n")
    info_text.insert(tk.END, f"Number of columns: {len(df.columns)}\n")
    info_text.insert(tk.END, f"Data types:\n{df.dtypes}\n")
    info_text.insert(tk.END, f"Summary statistics:\n{df.describe()}\n")
    info_text.configure(state="disabled")

#vẽ scatter
def scatter(columns):
    for widget in tab8.winfo_children():
        widget.destroy()
    bien1 = ttk.Combobox(tab8, state="readonly")
    bien1.grid(row=0, column=1, padx=10, pady=10)
    
    bien2 = ttk.Combobox(tab8, state="readonly")
    bien2.grid(row=2, column=1, padx=10, pady=5)
    
    # Gán giá trị cho combobox chỉ khi cột không phải là kiểu object
    numerical_columns = [col for col in columns if df[col].dtype != "object"]
    bien1['values'] = numerical_columns
    bien2['values'] = numerical_columns
    
    def plot_scatter():
        col1 = bien1.get()
        col2 = bien2.get()
        
        if col1 and col2:
            plt.figure(figsize=(7, 4))
            plt.scatter(df[col1], df[col2])
            plt.title(f'Scatter plot between {col1} and {col2}')
            plt.xlabel(col1)
            plt.ylabel(col2)
            # Tạo FigureCanvasTkAgg và hiển thị trên cửa sổ mới
        canvas = FigureCanvasTkAgg(plt.gcf(), master=tab8)
        canvas.draw()
        canvas.get_tk_widget().grid(row=4, column=1, sticky="nsew")
    btn_scatter = ttk.Button(tab8, text="Vẽ Scatter", command=plot_scatter)
    btn_scatter.grid(row=3, column=1, columnspan=2, pady=5)

# Tìm kiếm trường tương tự
def find_similar_products():
    try:
        selected_indices = product_description_listbox.curselection()
        if not selected_indices:
            messagebox.showerror("Error", "Vui lòng chọn một hoặc nhiều cột để mô tả sản phẩm.")
            return
        
        selected_columns = [product_description_listbox.get(i) for i in selected_indices]
        display_variable = display_variable_combobox.get()
        
        if not display_variable:
            messagebox.showerror("Error", "Vui lòng chọn một biến để hiển thị trường của biến đó.")
            return
        
        if not all(col in df.columns for col in selected_columns + [display_variable]):
            messagebox.showerror("Error", "Một hoặc nhiều cột được chọn không tồn tại trong tập dữ liệu.")
            return
        
        df[selected_columns] = df[selected_columns].astype(str)
        product_descriptions = df[selected_columns].agg(' '.join, axis=1)
        
        tfidf_vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = tfidf_vectorizer.fit_transform(product_descriptions)
        
        feature_names = tfidf_vectorizer.get_feature_names_out()

        cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
        
        # Plot a subset of the Cosine Similarity Matrix as a Heatmap
        subset_size = 50
        cosine_sim_subset = cosine_sim[:subset_size, :subset_size]
        
        # Plot Cosine Similarity Matrix
        fig, ax = plt.subplots(figsize=(4.33, 3.67))  # You can adjust the figure size as needed
        sns.heatmap(cosine_sim_subset, cmap='viridis', ax=ax)
        ax.set_title('Cosine Similarity Matrix')
        ax.set_xlabel('Products')
        ax.set_ylabel('Products')
        
        canvas = FigureCanvasTkAgg(fig, master=tab6)
        canvas.draw()
        canvas.get_tk_widget().grid(row=4, column=2, sticky="nsew", padx=5, pady=15)
        
        similar_products = []
        for idx, row in df.iterrows():
            similar_indices = cosine_sim[idx].argsort()[:-5:-1]
            for i in similar_indices:
                if i != idx:
                    similar_products.append({
                        'Product': row.get(display_variable, 'N/A'),
                        'Similar Product': df.iloc[i].get(display_variable, 'N/A'),
                        'Score': cosine_sim[idx][i]
                    })
                    
        similar_products.sort(key=lambda x: x['Score'], reverse=True)
        display_similar_products(similar_products)
    
    except Exception as e:
        messagebox.showerror("Error", str(e))

# Hiển thị kết quả trường tương tự
def display_similar_products(similar_products):
    for widget in similar_products_frame.winfo_children():
        widget.destroy()
    
    cols = ('Product', 'Similar Product', 'Score')
    tree = ttk.Treeview(tab6, columns=cols, show='headings')
    
    for col in cols:
        tree.heading(col, text=col)
        tree.column(col, width=200 if col != 'Score' else 100, anchor=tk.CENTER)
    
    for item in similar_products:
        tree.insert("", "end", values=(item['Product'], item['Similar Product'], f"{item['Score']:.2f}"))
    
    tree.grid(row=4, column=0, sticky='nsew')
    
    # Add a scrollbar for the treeview
    scrollbar = ttk.Scrollbar(tab6, orient="vertical", command=tree.yview)
    tree.configure(yscrollcommand=scrollbar.set)
    scrollbar.grid(row=0, column=1, sticky='ns')
    
values = []
labels=[]
scale = None
target_encode = None
vars_encode = []
next_row = 0
linear = False

#Lấy giá trị được nhập để predict
def get_input_values():
    ip_data={}
    for value, label in zip(values, labels):
        v = value.get()
        l = label.cget("text")
        if df[l].dtype == 'object':
            v = str(v)
            if l in vars_encode:
                v = vars_encode[l].transform([v])[0]
        else:
            try:
                v = float(v)
            except ValueError:
                v = None 
        print(type(v))
        l = label.cget("text")
        print(l)
      #  print(v)
        ip_data[l] = v
    ip_df=pd.DataFrame([ip_data])
    
    numeric_cols = [col for col in ip_df.columns if ip_df[col].dtype != 'object']
    object_cols = [col for col in ip_df.columns if ip_df[col].dtype == 'object']
    ip_df = ip_df[numeric_cols + object_cols]
    i=0
    for col in ip_df.columns:
        
        if ip_df[col].dtype=="object":
            ip_df[col]= vars_encode[i].transform(ip_df[col])
            print("re: ",vars_encode[i].inverse_transform(ip_df[col]))
            print("ob: ",ip_df[col].values)
            i=i+1
    
    ip_df = scale.transform(ip_df)
    y = model_predict.predict(ip_df)
    
    # Nếu là mô hình tuyến tính, làm tròn giá trị dự đoán
    if linear:
        y = [round(pred) for pred in y]
    if(target_encode != None):
        y = target_encode.inverse_transform(y)
    print(y)
    #text_pre = "Predict: "+y
    t_pre = tk.Label(tab4, text="Predict:", font=("Arial", 12, "bold"), bg="white")
    t_pre.grid(row=next_row, column=0, padx=10, pady=6, sticky="nsew")
    pre = tk.Label(tab4, text=y[0], font=("Arial", 12, "bold"), bg="white")
    pre.grid(row=next_row, column=1, padx=10, pady=6, sticky="nsew")

#Tạo ô nhập giá trị dự đoán
def predict():
    global next_row  
    next_row = 0
    print("OK")
    try:
        for widget in tab4.winfo_children():
            widget.destroy()
    except Exception as e:
        print(f"Error destroying widget: {e}")
    labels.clear()
    values.clear()
    selected_variables = selected_input_variables_listbox.get(0, tk.END)
    print(selected_variables)
    i = 0
    # Duyệt qua các phần tử được chọn và tạo các label tương ứng
    for select in selected_variables:
        # select = selected_input_variables_listbox.get(index)
        # print(select)
        bien_info = tk.Label(tab4, text=select, font=("Arial", 12, "bold"), bg="white")
        bien_info.grid(row=i+1, column=0, padx=10, pady=6, sticky="nsew")
        labels.append(bien_info)
        entry_value="entry"+f"{i}"
        entry_value = tk.Entry(tab4, font=("Arial", 12))
        #entry_value = tk.Text(tab4, height = 3, width = 10) 
        entry_value.grid(row=i+1, column=1, padx=10, pady=6, sticky="nsew")
        # values.append(entry_value.get(1.0, "end-1c"))
        values.append(entry_value)
        print(entry_value)
        i=i+1
    
    get_values_button = tk.Button(tab4, text="Lấy giá trị", command=get_input_values)
    get_values_button.grid(row=len(variables)+1, column=0, columnspan=2, padx=10, pady=10, sticky="nsew")
    next_row = len(variables)+2

#Các kiểu fill missing value
def fill_missing_value(df, column, fill_method):
    if fill_method == "Median Value":
        fill_value = df[column].median()  # Lấy giá trị xuất hiện nhiều nhất
    elif fill_method == "Mode Value":
        fill_value = df[column].mode().iloc[0]  # Lấy giá trị mode
    elif fill_method == "Mean Value":
        fill_value = df[column].mean()  # Lấy giá trị trung bình
    elif fill_method == "Mode Imputation":
        fill_value = df[column].mode().iloc[0]
    df[column].fillna(fill_value, inplace=True)
    print(f"Filled missing values for column '{column}' with '{fill_method}': {fill_value}")
    missingvalue()
    load_dataset()
    
#lưu File
def save(df):
    # Mở hộp thoại cho phép người dùng chọn vị trí và tên file CSV để lưu
    filepath = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")])
    
    if filepath:
        try:
            df.to_csv(filepath, index=False)
            print("Lưu thành công vào", filepath)
        except Exception as e:
            print("Lưu không thành công. Lỗi:", str(e))
    else:
        print("Không chọn đường dẫn lưu tệp.")
        
#gọi hàm lưu file
def on_save():
    save(df)
 
#Hiển thị giá trị missing value   
def display_missing_info(df, tab):
    # Hiển thị thông tin của dataset ở panel2
    for widget in tab3.winfo_children():
        widget.destroy()
        
    save_button = ttk.Button(tab, text="Save", command=lambda: on_save())
    save_button.pack(side="bottom", padx=10, pady=5)
    # Tạo Canvas để cho phép cuộn
    canvas = tk.Canvas(tab)
    canvas.pack(side="left", fill="both", expand=True)

    # Tạo Frame con để chứa các dòng thông tin
    frame = tk.Frame(canvas)
    canvas.create_window((0, 0), window=frame, anchor="nw")

    for i, column in enumerate(df.columns):
        # Tính phần trăm thiếu của cột
        missing_percentage = (df[column].isnull().sum() / len(df)) * 100
        
        # Tạo Frame con cho mỗi cột
        column_frame = tk.Frame(frame)
        column_frame.pack(fill="x", padx=10, pady=5)

        # Hiển thị tên cột và phần trăm thiếu
        label_name = tk.Label(column_frame, text=f"Column Name: {column}")
        label_name.pack(side="left", padx=(0, 10))

        # Tạo Combobox cho cách điền giá trị thiếu
        if df[column].dtype == "object":
            fill_combobox = ttk.Combobox(column_frame, values=["Mode Imputation"], state="readonly")
        else:
            fill_combobox = ttk.Combobox(column_frame, values=["Median Value","Mode Value", "Mean Value"], state="readonly")
                
        # Tạo Button để điền giá trị
        fill_button = ttk.Button(column_frame, text="Fill", command=lambda col=column, method=fill_combobox: fill_missing_value(df, col, method.get()))
        fill_button.pack(side="right", padx=(0, 10))
        fill_combobox.pack(side="right", padx=(0, 10))
        
        label_col = tk.Label(column_frame, text=f"Total rows: {df[column].count()}") #total rows have value
        label_col.pack(side="right", padx=(0, 10))

        label_missing = tk.Label(column_frame, text=f"Missing Percentage: {missing_percentage:.2f}%")
        label_missing.pack(side="right", padx=(0, 10))

        if missing_percentage == 0:
            fill_combobox.pack_forget()
            # label_missing.pack_forget()
            fill_button.pack_forget()

    # Kích hoạt cuộn Canvas
    canvas.update_idletasks()
    canvas.config(scrollregion=canvas.bbox("all"))
 
# Gọi hiển thị    
def missingvalue():
    display_missing_info(df, tab3)

#Thêm cột df cần tham gia để dự đoán
def add_input_variable():
    selected_indices = input_variable_listbox.curselection()
    for idx in selected_indices[::-1]:  
        variable = input_variable_listbox.get(idx)
        input_variable_listbox.delete(idx)
        selected_input_variables_listbox.insert(tk.END, variable)
        selected_input_variables.append(variable)
       
#Xóa cột df 
def remove_input_variable():
    selected_indices = selected_input_variables_listbox.curselection()
    for idx in selected_indices[::-1]: 
        variable = selected_input_variables_listbox.get(idx)
        selected_input_variables_listbox.delete(idx)
        input_variable_listbox.insert(tk.END, variable)
        selected_input_variables.remove(variable)
     
#Chọn model   
def get_model():
    selected_value = model_combobox.get()
    return selected_value

#Chọn biến mục tiêu
def get_taget():
    selected_value = target_variable_combobox.get()
    return selected_value

#Ma trận confusion
def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.set_title(title)
    plt.colorbar(im, ax=ax)
    tick_marks = np.arange(len(classes))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(classes, rotation=45)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        ax.text(j, i, format(cm[i, j], fmt),
                 ha="center", va="center",
                 color="white" if cm[i, j] > thresh else "black")

    ax.set_ylabel('True label')
    ax.set_xlabel('Predicted label')

    return fig

def plot_correlation_matrix():
    # Tạo cửa sổ mới
    for widget in tab5.winfo_children():
        widget.destroy()
    # Chọn các cột dạng số từ DataFrame
    numeric_df = df.select_dtypes(include=['int', 'float'])

    # Tính ma trận tương quan
    correlation_matrix = numeric_df.corr()

    # Vẽ heatmap cho ma trận tương quan
    plt.figure(figsize=(8, 6))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title("Correlation Matrix")

    # Tạo FigureCanvasTkAgg và hiển thị trên cửa sổ mới
    canvas = FigureCanvasTkAgg(plt.gcf(), master=tab5)
    canvas.draw()
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

#show kết quả dự đoán
def show_model_result(model_name, result_text,pt, y_test, y_pred):
    # Xóa tất cả các widget trong panel2 trước khi hiển thị kết quả mới
    for widget in tab2.winfo_children():
        widget.destroy()
    result_label = tk.Label(tab2, text=model_name, font=("Tahoma", 12))
    result_label.grid(row=1, column=0)  
    # Tạo label để hiển thị kết quả
    result_label = tk.Label(tab2, text=result_text, font=("Tahoma", 12))
    result_label.grid(row=2, column=0)  # Đặt label ở hàng 1, cột 0 trong 
    
    result_label = tk.Label(tab2, text=pt, font=("Tahoma", 12))
    result_label.grid(row=3, column=0,  sticky="nsew")  # Đặt label ở hàng 1, cột 0 trong 
    #========================
    print(y_test)
    print()
    print(y_pred)

#định nghĩa text pt
def pt(intercept,coefficients,X,y):
    feature_names = list(X.columns)
   # namey = y.columns.tolist()[0]

    pt = f"{y} = {intercept:.4f}"
    count =0
    for feature_name, coef in zip(feature_names, coefficients):
        count+=1
        if count%3==0:
            pt+="\n"
        pt += f" + {feature_name} * {coef:.4f}"
    print("Biểu thức dự đoán:", pt)
    return pt

    #========================

#Xử lí dự đoán
def show_new_page(): 
    global df
    global model_predict, scale, target_encode, vars_encode, linear
    linear = False
        # Khai báo các biến toàn cục
    model_predict = None
    scale = None
    target_encode = None
    vars_encode = []
    df_new = df.copy()
    df_new.dropna(inplace=True)
    print(df_new.info())
    categories=[]
    nums=[]
    selected_variables = selected_input_variables_listbox.get(0, tk.END)
    for variable in selected_variables:
        if(df_new[variable].dtypes == "object"):
            categories.append(variable)
        else:
            nums.append(variable)
            
    df_numericals = df_new[nums]
    # for num in nums:
    #     mean = df_numericals[num].mean()
    #     df_numericals[num].fillna(mean, inplace=True) 
    if categories:
        # imputer = SimpleImputer(strategy='most_frequent')
        df_categories = df_new[categories]
        # df_categories = pd.DataFrame(imputer.fit_transform(df_categories), columns=df_categories.columns)
    
        print(df_categories.info())
    
        for cate in categories:
            lb_encoder = LabelEncoder()
            #df_categories[cate] = lb_encoder.fit_transform(df_new[cate])
            # df_categories.loc[:, cate] = lb_encoder.fit_transform(df_new[cate])
            df_categories.loc[:, cate] = lb_encoder.fit_transform(df_new[cate])
            vars_encode.append(lb_encoder)

        X = pd.concat([df_numericals, df_categories], axis=1)
    else:
        X=df_numericals
        
    print(X.info())
    
    taget = get_taget()
    if(df_new[taget].dtype=="object"):
        # imputer_tg = SimpleImputer(strategy='most_frequent')
        # imputed_values = imputer_tg.fit_transform(df[[taget]])
        # df[taget] = pd.DataFrame(imputed_values, columns=df[[taget]].columns)
        
        lb_encoder = LabelEncoder()
        df_new[taget]=lb_encoder.fit_transform(df_new[taget])
        y=df_new[taget]
        # y = pd.DataFrame(y_encoded, columns=[df[taget].name])
        target_encode = lb_encoder
    else:
        # mean_taget = df[taget].mean()
        # df[taget].fillna(mean_taget, inplace=True)
        y=df_new[taget]
    X_train, X_test, y_train, y_test = train_test_split(X, y , test_size=0.2, random_state=0)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    scale = scaler
    print("Số lượng mẫu trong tập huấn luyện:", X_train.shape[0])
    print("Số lượng mẫu trong tập kiểm tra:", X_test.shape[0])
    model = get_model()
    
    if(model=="Linear Regression"):
        ln = LinearRegression()
        ln.fit(X_train, y_train)
        model_predict = ln
        linear = True
        y_pred = ln.predict(X_test)
   
        intercept , coef = ln.intercept_, ln.coef_
        t_pt = pt(intercept, coef,X, y.name)
        rmse = r2_score(y_test, y_pred)
        rmse_text = f"Root Mean Squared Error (RMSE): {rmse:.2f}"
        show_model_result("Linear Regression", rmse_text,t_pt, y_test, y_pred)
        
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111)

        ax.scatter(y_test, y_pred, color='blue', label='Actual vs. Predicted')
        ax.plot(y_test, y_test, color='red', label='Perfect Prediction')
        ax.set_title('Actual vs. Predicted Values')
        ax.set_xlabel('Actual Values')
        ax.set_ylabel('Predicted Values')
        ax.legend()

        # Tạo một FigureCanvasTkAgg và pack nó vào panel2
        canvas = FigureCanvasTkAgg(fig, master=tab2)
        canvas.draw()
        canvas.get_tk_widget().grid(row=1, column=0, sticky="nsew")
        
    if(model=="Logistic Regression"):
        clf = LogisticRegression()
        clf.fit(X_train, y_train)
        model_predict = clf
        y_pred = clf.predict(X_test)
        print(y_pred)
        print(y_test)
        unique_values = np.unique(y_pred)
        unique_count = len(unique_values)
        print("Number of unique values in 'predict':", unique_count)
        intercept , coef = clf.intercept_[0], clf.coef_[0]
        t_pt = pt(intercept, coef,X, y.name)
        accuracy = clf.score(X_test, y_test)
        print("Do chinh xac: ", accuracy)
        accuracy_text = f"Độ chính xác (Accuracy): {accuracy:.2f}"
        show_model_result("Logistic Regression", accuracy_text,t_pt, y_test, y_pred)
        
        y_pred_proba = clf.predict_proba(X_test)[:, 1]  
        #X_test_1d = X_test.ravel()
        pca = PCA(n_components=1)
        X_test_1d = pca.fit_transform(X_test) 
        
        min_size = min(len(X_test_1d), len(y_pred_proba))
        X_test_1d = X_test_1d[:min_size]
        y_pred_proba = y_pred_proba[:min_size]
        
        def sigmoid(x):
            return 1 / (1 + np.exp(-x))
        x_values = np.linspace(-10, 10, 100)

        sigmoid_values = sigmoid(x_values)
        sigmoid_values = sigmoid_values[:len(x_values)]

        plt.figure(figsize=(8, 6))
        # plt.scatter(X_test_1d, y_test, color='blue', label='Logistic Regression')
        # Vẽ dữ liệu và đường phân cách
        plt.scatter(X_test_1d, y_test, color='blue', label='Test Data')

        # plt.plot(X_test_1d, np.where(y_pred_proba >= 0.5, 1, 0), color='red', linestyle='--', label='Decision Boundary')
        plt.plot(x_values, sigmoid_values, color='red', label='Sigmoid Function')
        plt.xlabel('Input Variable')
        plt.ylabel('Probability of Positive Class')
        plt.title('Logistic Regression')
        plt.legend()
        plt.grid(True)

        canvas = FigureCanvasTkAgg(plt.gcf(), master=tab2)
        canvas.draw()
        canvas.get_tk_widget().grid(row=1, column=0, sticky="nsew")

    if(model=="KNN"):
        k = 5 
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)
        # Nếu mô hình đã được đào tạo, không cần gán lại
        model_predict = knn
        y_pred = knn.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        t_pt ="noooo"
        accuracy_text = f"Độ chính xác (Accuracy): {accuracy:.2f}"
        show_model_result("KNN", accuracy_text,t_pt, y_test, y_pred)
        
        pca = PCA(n_components=2)
        X_train_pca = pca.fit_transform(X_train)
        X_test_pca = pca.transform(X_test)
        
        knn1 = KNeighborsClassifier(n_neighbors=k)
        knn1.fit(X_train_pca, y_train)
        y_pred = knn1.predict(X_test_pca)

        cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
        plt.figure(figsize=(8, 6))
        plt.scatter(X_test_pca[:, 0], X_test_pca[:, 1], c=y_test, cmap=cmap_light, edgecolor='k', s=100)
        plt.title("KNN Decision Boundary (PCA)")
        plt.xlabel("Principal Component 1")
        plt.ylabel("Principal Component 2")

        # Vẽ vùng quyết định của mô hình KNN
        x_min, x_max = X_test_pca[:, 0].min() - 1, X_test_pca[:, 0].max() + 1
        y_min, y_max = X_test_pca[:, 1].min() - 1, X_test_pca[:, 1].max() + 1
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
        Z = knn1.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        plt.contourf(xx, yy, Z, alpha=0.3, cmap=cmap_light)
        plt.colorbar()
        
        canvas = FigureCanvasTkAgg(plt.gcf(), master=tab2)
        canvas.draw()
        canvas.get_tk_widget().grid(row=1, column=0, sticky="nsew") 
        
    if(model=="SVM"):
        #svc = SVC(kernel='linear', probability=True)
        svc=SVC(kernel='linear', C=1.0, random_state=42)
        svc.fit(X_train, y_train)
        model_predict = svc
        y_pred = svc.predict(X_test)
        print(y_pred)
        print(y_test)
        unique_values = np.unique(y_pred)
        unique_count = len(unique_values)
        print("Number of unique values in 'predict':", unique_count)
        intercept , coef = svc.intercept_[0], svc.coef_[0]
        t_pt = pt(intercept, coef,X, y.name)
        accuracy =  accuracy_score(y_test, y_pred)
        print("Do chinh xac: ", accuracy)
        accuracy_text = f"Độ chính xác (Accuracy): {accuracy:.2f}"
        show_model_result("SVM", accuracy_text,t_pt, y_test, y_pred)
        # Giảm chiều dữ liệu xuống còn 2 chiều bằng PCA để vẽ biểu đồ
        svc1=SVC(kernel='linear', C=1.0, random_state=42)
        pca = PCA(n_components=2)
        X_train_pca = pca.fit_transform(X_train)
        X_test_pca = pca.transform(X_test)
        
        # Huấn luyện mô hình KNN trên dữ liệu đã giảm chiều
        svc1.fit(X_train_pca, y_train)
        y_pred = svc1.predict(X_test_pca)
        
        # Xác định giới hạn của biểu đồ dựa trên dữ liệu giảm chiều
        x_min, x_max = X_test_pca[:, 0].min() - 1, X_test_pca[:, 0].max() + 1
        y_min, y_max = X_test_pca[:, 1].min() - 1, X_test_pca[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))
        Z = svc1.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        plt.figure(figsize=(8, 6))
        plt.contourf(xx, yy, Z, alpha=0.8)
        plt.scatter(X_test_pca[:, 0], X_test_pca[:, 1], c=y_test, cmap=plt.cm.Paired)

        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.title('SVM Decision Boundary')

        canvas = FigureCanvasTkAgg(plt.gcf(), master=tab2)
        canvas.draw()
        canvas.get_tk_widget().grid(row=1, column=0, sticky="nsew")
        
    if(model=="K-means"):
        for widget in tab2.winfo_children():
            widget.destroy()
        #data = pd.concat([X, y], axis=1)
        data = X
        pca = PCA(n_components=2)
        data = pca.fit_transform(data)
        so_cum = y.nunique()
        print(so_cum)
        kmeans = KMeans(n_clusters=so_cum, random_state=42)
        kmeans.fit(data)
        kmeans1 = KMeans(n_clusters=so_cum, random_state=42)
        kmeans1.fit(X)
        model_predict = kmeans1
        labels = kmeans.labels_
        centroids = kmeans.cluster_centers_

        # Vẽ biểu đồ các cụm dữ liệu
        plt.figure(figsize=(8, 6))

        plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis', alpha=0.8, edgecolors='k')
        plt.scatter(centroids[:, 0], centroids[:, 1], c='red', s=200, alpha=0.8, edgecolors='k', label='Centroids')

        plt.xlabel('chiều 1')
        plt.ylabel('chiều 2')
        plt.title('K-means Clustering of Iris Dataset')
        plt.legend()
               
        canvas = FigureCanvasTkAgg(plt.gcf(), master=tab2)
        canvas.draw()
        canvas.get_tk_widget().grid(row=1, column=0, sticky="nsew")
        
    if(model == "Decision Tree") :
        from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
        from sklearn.experimental import enable_halving_search_cv
        from sklearn.model_selection import HalvingGridSearchCV
        from sklearn.tree import DecisionTreeClassifier
        from sklearn import tree

        for widget in tab2.winfo_children():
            widget.destroy()

        clf = tree.DecisionTreeClassifier(random_state=42, max_depth=5)
        clf.fit(X_train, y_train)
        model_predict = clf
                # Cắt tỉa sau sử dụng cost complexity pruning
        path = clf.cost_complexity_pruning_path(X_train, y_train)
        ccp_alphas, impurities = path.ccp_alphas, path.impurities

        # Tìm giá trị alpha tốt nhất dựa trên cross-validation
        dct = HalvingGridSearchCV(clf, {"ccp_alpha": ccp_alphas[:-1]}, cv=5)
        dct.fit(X_train, y_train)
        best_model = dct.best_estimator_
        # Dự đoán nhãn của tập kiểm tra
        y_pred = dct.predict(X_test)
        
        # Tính độ chính xác của mô hình
        accuracy = accuracy_score(y_test, y_pred)
        print("Độ chính xác của mô hình là:", accuracy)
        t_pt="Noo"
        accuracy_text = f"Độ chính xác (Accuracy): {accuracy:.2f}"
        show_model_result("Decision_tree", accuracy_text,t_pt, y_test, y_pred)
        
        #plt.figure(figsize = (8,6))
        #tree.plot_tree(dct, fontsize = 8,rounded = True , filled = True)
        fig, ax = plt.subplots(figsize=(8, 6))
        class_names = [str(class_name) for class_name in dct.classes_]
        tree.plot_tree(best_model,fontsize=6 ,filled=True,feature_names=X.columns, class_names=class_names)

        canvas = FigureCanvasTkAgg(plt.gcf(), master=tab2)
        canvas.draw()
        canvas.get_tk_widget().grid(row=1, column=0, sticky="nsew")
    
    if model == "Random Forest":
        for widget in tab2.winfo_children():
            widget.destroy()
            
        classifier = RandomForestClassifier(n_estimators=10, criterion="entropy")  
        classifier.fit(X_train, y_train)
        model_predict = classifier
        y_pred = classifier.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        accuracy_label = Label(tab2, text=f"Accuracy: {accuracy:.2f}", font=("Arial", 12))
        accuracy_label.grid(row=0, column=0, padx=10, pady=10)
        
        pca = PCA(n_components=2)
        X_train_pca = pca.fit_transform(X_train)
        X_test_pca = pca.transform(X_test)
        
        classifier1 = RandomForestClassifier(n_estimators=10, criterion="entropy")  
        classifier1.fit(X_train_pca, y_train)
        y_pred = classifier1.predict(X_test_pca)

        x_set, y_set = X_train_pca, y_train
        x1, x2 = np.meshgrid(np.arange(start=x_set[:, 0].min() - 1, stop=x_set[:, 0].max() + 1, step=0.01),
                            np.arange(start=x_set[:, 1].min() - 1, stop=x_set[:, 1].max() + 1, step=0.01))  
        plt.figure(figsize=(8, 6))
        plt.contourf(x1, x2, classifier1.predict(np.array([x1.ravel(), x2.ravel()]).T).reshape(x1.shape),  
                    alpha=0.75, cmap=ListedColormap(('purple', 'green')))  
        plt.xlim(x1.min(), x1.max())  
        plt.ylim(x2.min(), x2.max())  
        for i, j in enumerate(np.unique(y_set)):  
            plt.scatter(x_set[y_set == j, 0], x_set[y_set == j, 1],  
                        color=ListedColormap(('purple', 'green'))(i), label=j)  # Fix the scatter function call
        plt.title('Random Forest Algorithm (Training set)')  
        plt.xlabel("Principal Component 1") 
        plt.ylabel("Principal Component 2") 
        plt.legend()  

        canvas = FigureCanvasTkAgg(plt.gcf(), master=tab2)
        canvas.draw()
        canvas.get_tk_widget().grid(row=1, column=0, sticky="nsew")

    predict()

variables = []
selected_input_variables = []

#===============================
#Hàm chức năng của menu
def toggle_panel1():
    #predict()
    if panel1_visible.get():
        panel1.grid_remove()
        panel1_visible.set(False)
    else:
        panel1.grid()
        panel1_visible.set(True)

root = tk.Tk()
root.title("CSV Variable Selector")
root.geometry("1800x850")

panel1_visible = tk.BooleanVar(value=True)

# Panel 1
panel1 = tk.Frame(root, width=350, height=550, borderwidth=1, relief="ridge")
panel1.grid(row=0, column=0, padx=20, pady=30)

style = ttk.Style()

style.configure("Panel1.TFrame", background="powderblue")

lbl_info_header = tk.Label(panel1, text="Information", font=("Arial", 13, "bold"), background="powderblue")
lbl_info_header.grid(row=0, column=0, columnspan=3, padx=10, sticky="nsew")

# Load menu icon image
menu_icon = tk.PhotoImage(file="./app/img/menu.png").subsample(6)
menu_button = ttk.Button(root, image=menu_icon, command=toggle_panel1)
# menu_button = ttk.Button(root,text="Menu", command=toggle_panel1)
# menu_button.grid(row=0, column=0, padx=10, pady=10, sticky="nw")

lbl_select_file = ttk.Label(panel1, text="Select File CSV:", font=("Arial", 10, "bold"), background="#f0f0f0")
lbl_select_file.grid(row=1, column=0, sticky="nsew", padx=10, pady=20)
load_button = ttk.Button(panel1, text="Load CSV File", command=load_csv_variables)
load_button.grid(row=1, column=1)

select_label = ttk.Label(panel1, text="In4 for each variable:", font=("Arial", 10, "bold"), background="#f0f0f0")
select_label.grid(row=2, column=0, sticky="nsew", pady=10, padx=10)
select_variable_combobox = ttk.Combobox(panel1, state="readonly")
select_variable_combobox.bind("<<ComboboxSelected>>", lambda event: load_variable_info(select_variable_combobox.get()))
select_variable_combobox.grid(row=2, column=1, padx=10)

target_label = ttk.Label(panel1, text="Select Target Variable:", font=("Arial", 10, "bold"), background="#f0f0f0")
target_label.grid(row=3, column=0, sticky="nsew", pady=10, padx=10)
target_variable_combobox = ttk.Combobox(panel1, state="readonly")
target_variable_combobox.grid(row=3, column=1, padx=10)

input_label = ttk.Label(panel1, text="Select Input Variables:", font=("Arial", 10, "bold"), background="#f0f0f0")
input_label.grid(row=4, column=0, sticky="nsew", pady=10, padx=10)
input_variable_listbox = tk.Listbox(panel1, selectmode=tk.MULTIPLE, height=8)
input_variable_listbox.grid(row=4, column=1, pady=10)
add_button = ttk.Button(panel1, text="Add", command=add_input_variable)
add_button.grid(row=4, column=2, padx=10)

selected_label = ttk.Label(panel1, text="Selected Input Variables:", font=("Arial", 10, "bold"), background="#f0f0f0")
selected_label.grid(row=5, column=0, sticky="nsew", pady=10, padx=10)
selected_input_variables_listbox = tk.Listbox(panel1, selectmode=tk.MULTIPLE, height=8)
selected_input_variables_listbox.grid(row=5, column=1, pady=10)
remove_button = ttk.Button(panel1, text="Remove", command=remove_input_variable)
remove_button.grid(row=5, column=2, padx=10)

status_label = ttk.Label(panel1, text="Chọn Model:", font=("Arial", 10, "bold"), background="#f0f0f0")
status_label.grid(row=6, column=0, sticky="nsew", pady=10, padx=10)
model_combobox = ttk.Combobox(panel1, values=["Linear Regression", "Logistic Regression", "Decision Tree", "KNN","K-means","SVM", "Random Forest"], state="readonly")
model_combobox.grid(row=6, column=1, pady=10)
add_button = ttk.Button(panel1, text="Execution", command=show_new_page)
add_button.grid(row=7, column=1, pady=20)

# Panel 2
panel2 = tk.Frame(root, width=400, height=550, relief="ridge")
panel2.grid(row=0, column=1, padx=20, pady=20)

style.theme_use('clam')  # Chọn một style ('clam', 'default', 'alt', 'classic')

# Thêm Tab menu
result_notebook = ttk.Notebook(panel2, style="TNotebook")
result_notebook.grid(row=1, column=0, columnspan=2, padx=10, pady=10, sticky="nsew")

# Tab 1
tab1 = tk.Frame(result_notebook, bg="white")
result_notebook.add(tab1, text="Dataset's info")

tab1_info_label = tk.Label(tab1, text="Review information of dataset", font=("Arial", 12, "italic"), bg="white")
tab1_info_label.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")

# Tab 2
tab2 = tk.Frame(result_notebook, bg="white")
result_notebook.add(tab2, text="Chart")

tab2_label = tk.Label(tab2, text="Show model's chart", font=("Arial", 12, "italic"), bg="white")
tab2_label.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")

style.configure("TNotebook.Tab", font=("Arial", 12), borderwidth=2, foreground="black")
style.map("TNotebook.Tab", background=[("selected", "powderblue")], relief=[("selected", "ridge")])

# Tab 3
tab3 = tk.Frame(result_notebook, bg="white")
result_notebook.add(tab3, text="Data processing")

# Tab 4
tab4 = tk.Frame(result_notebook, bg="white")
result_notebook.add(tab4, text="Predict")

# Tab 5
tab5 = tk.Frame(result_notebook, bg="white")
result_notebook.add(tab5, text="Correlate")

# Tab 6
tab6 = ttk.Frame(result_notebook)
result_notebook.add(tab6, text='Similar Products')
similar_products_frame = ttk.Frame(tab6)
similar_products_frame.grid(row=0, column=0, padx=10, pady=10, sticky='nsew')

# Listbox để chọn cột mô tả sản phẩm
product_description_label = ttk.Label(tab6, text="Chọn Cột Mô Tả Sản Phẩm:")
product_description_label.grid(row=0, column=0, padx=5, pady=5, sticky='w')
product_description_listbox = tk.Listbox(tab6, selectmode=tk.MULTIPLE, exportselection=False)
product_description_listbox.grid(row=1, column=0, padx=5, pady=5, sticky='w')
listbox_scrollbar = ttk.Scrollbar(tab6, orient="vertical", command=product_description_listbox.yview)
listbox_scrollbar.grid(row=1, column=1, sticky='ns')
product_description_listbox.config(yscrollcommand=listbox_scrollbar.set)

# Combobox để chọn biến hiển thị trong bảng kết quả
display_variable_label = ttk.Label(tab6, text="Chọn Biến Để Hiển Thị:")
display_variable_label.grid(row=0, column=2, padx=5, pady=5, sticky='w')
display_variable_combobox = ttk.Combobox(tab6, state="readonly")
display_variable_combobox.grid(row=0, column=3, padx=5, pady=5, sticky='w')

find_button = ttk.Button(tab6, text="Tìm Sản Phẩm Tương Tự", command=find_similar_products)
find_button.grid(row=3, column=0, columnspan=2, pady=10)

# Tab 7
tab7 = tk.Frame(result_notebook, bg="white")
result_notebook.add(tab7, text="Graph each variable")

# Tab 8
tab8 = tk.Frame(result_notebook, bg="white")
result_notebook.add(tab8, text="Scatter")

root.mainloop()
