import numpy as np
from keras.layers import Conv2D, Input, BatchNormalization, LeakyReLU, ZeroPadding2D, UpSampling2D # type: ignore
from keras.layers.merge import add, concatenate # type: ignore
from keras.models import Model # type: ignore
import struct
import cv2

#
class WeightReader:
    # Phương thức khởi tạo nhận tham số weight_file là tên tệp nhị phân chứa trọng số
    def __init__(self, weight_file):
        # Mở tệp nhị phân chứa trọng số ở chế độ đọc ('rb')
        with open(weight_file, 'rb') as w_f:
            # Đọc phiên bản chính, phụ và bản sửa đổi (12 byte đầu tiên)
            # Mỗi lần đọc 4 byte và giải mã thành số nguyên ('i')
            major, = struct.unpack('i', w_f.read(4))
            minor, = struct.unpack('i', w_f.read(4))
            revision, = struct.unpack('i', w_f.read(4))
            
            # Xác định định dạng của tệp dựa trên phiên bản đọc được
            if (major * 10 + minor) >= 2 and major < 1000 and minor < 1000:
                w_f.read(8)  # Đọc thêm 8 byte nếu phiên bản >= 2
            else:
                w_f.read(4)  # Đọc thêm 4 byte cho các phiên bản khác
            
            # Xác định cờ hoán vị (transpose) dựa trên phiên bản tệp
            transpose = (major > 1000) or (minor < 1000)
            
            # Đọc phần còn lại của tệp chứa dữ liệu trọng số
            binary = w_f.read()                                                                                                       
        
        # Lưu trữ toàn bộ trọng số vào mảng NumPy
        self.offset = 0
        self.all_weights = np.frombuffer(binary, dtype='float32')
    
    # Phương thức đọc một lượng trọng số xác định từ mảng
    def read_bytes(self, size):
        # Cập nhật vị trí đọc tiếp theo trong mảng
        self.offset += size
        # Trả về mảng trọng số từ vị trí hiện tại
        return self.all_weights[self.offset - size:self.offset]
    
    # Phương thức tải trọng số vào mô hình đã cho
    def load_weights(self, model):
        # Lặp qua từng lớp convolution (từ 0 đến 105)
        for i in range(106):
            try:
                # Lấy lớp convolution từ mô hình theo tên ('conv_i')
                conv_layer = model.get_layer('conv_' + str(i))
                print("Loading weights of convolution #" + str(i))
                
                # Các lớp có số thứ tự không phải là 81, 93, 105 có thêm lớp chuẩn hóa
                if i not in [81, 93, 105]:
                    norm_layer = model.get_layer('bnorm_' + str(i))
                    # Lấy kích thước của trọng số lớp chuẩn hóa
                    size = np.prod(norm_layer.get_weights()[0].shape)
                    
                    # Đọc và thiết lập trọng số lớp chuẩn hóa: beta, gamma, mean, variance
                    beta = self.read_bytes(size)  # bias
                    gamma = self.read_bytes(size)  # scale
                    mean = self.read_bytes(size)  # mean
                    var = self.read_bytes(size)  # variance
                    norm_layer.set_weights([gamma, beta, mean, var])
                
                # Nếu lớp convolution có trọng số bias
                if len(conv_layer.get_weights()) > 1:
                    # Đọc trọng số bias và kernel
                    bias = self.read_bytes(np.prod(conv_layer.get_weights()[1].shape))
                    kernel = self.read_bytes(np.prod(conv_layer.get_weights()[0].shape))
                    
                    # Định dạng lại kernel (chuyển chiều) và thiết lập trọng số cho lớp
                    kernel = kernel.reshape(list(reversed(conv_layer.get_weights()[0].shape)))
                    kernel = kernel.transpose([2, 3, 1, 0])
                    conv_layer.set_weights([kernel, bias])
                else:
                    # Nếu không có bias, chỉ đọc trọng số kernel
                    kernel = self.read_bytes(np.prod(conv_layer.get_weights()[0].shape))    
                    kernel = kernel.reshape(list(reversed(conv_layer.get_weights()[0].shape)))
                    kernel = kernel.transpose([2, 3, 1, 0])
                    conv_layer.set_weights([kernel])
            except ValueError:
                # Nếu không tìm thấy lớp convolution, in thông báo lỗi
                print("No convolution #" + str(i))
                
    # Phương thức đặt lại vị trí đọc (offset) về 0
    def reset(self):
        self.offset = 0

#      
class BoundBox:
    
    #
    def __init__(self, xmin, ymin, xmax, ymax, objness = None, classes = None):
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax
        
        self.objness = objness
        self.classes = classes
        
        self.label = -1
        self.score = -1
        
    #
    def get_label(self):
        if self.label == -1:
            self.label = np.argmax(self.classes)
            
        return self.label
    
    # 
    def get_score(self):
        if self.score == -1:
            self.score = self.classes[self.get_label()]
        
        return self.score
    
#
def _conv_block(inp, convs, skip=True):
    x = inp
    count = 0
    
    for conv in convs:
        if count == (len(convs) -2 ) and skip:
            skip_connection = x
        count += 1
        
        if conv['stride'] > 1: x = ZeroPadding2D(((1,0), (1,0)))(x) # peculiar padding as darknet prefer left and top
        x = Conv2D(conv['filter'],
                   conv['kernel'],
                   strides=conv['stride'],
                   padding='valid' if conv['stride'] > 1 else 'same', # peculiar padding as darknet prefer left and top
                   name='conv_' + str(conv['layer_idx']),
                   use_bias=False if conv['bnorm'] else True)(x)
        if conv['bnorm']: x = BatchNormalization(epsilon=0.001, name='bnorm_' + str(conv['layer_idx']))(x)
        if conv['leaky']: x = LeakyReLU(alpha=0.1, name='leaky_' +str(conv['layer_idx']))(x)
    
    return add([skip_connection, x]) if skip else x

#
def _interval_overlap(interval_a, interval_b):
    x1, x2 = interval_a
    x3, x4 = interval_b
    
    if x3 < x1:
        if x4 < x1:
            return 0
        else:
            return min(x2, x4) - x1
    else:
        if x2 < x3:
            return 0
        else:
            return min(x2, x4) - x3
        
# 
def _sigmoid(x):
    return 1. / (1. + np.exp(-x))

#
def bbox_iou(box1, box2):
    intersect_w = _interval_overlap([box1.xmin, box1.xmax], [box2.xmin, box2.xmax])
    intersect_h = _interval_overlap([box1.xmin, box1.xmax], [box2.xmin, box2.xmax])
    
    intersect = intersect_w * intersect_h
    
    w1, h1 = box1.xmax - box1.xmin, box1.ymax - box1.ymin
    w2, h2 = box2.xmax - box2.xmin, box2.ymax - box2.ymin
    
    union = w1*h1 + w2*h2 - intersect
    
    return float(intersect) / union

#
def make_yolov3_model():
    input_image = Input(shape=(None, None, 3))
    
    # Layer 0 => 4
    x = _conv_block(input_image, [{'filter': 32, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 0},
                                  {'filter': 64, 'kernel': 3, 'stride': 2, 'bnorm': True, 'leaky': True, 'layer_idx': 1},
                                  {'filter': 32, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 2},
                                  {'filter': 64, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 3}])
    
    # Layer 5 => 8
     x = _conv_block(input_image, [{'filter': 128, 'kernel': 3, 'stride': 2, 'bnorm': True, 'leaky': True, 'layer_idx': 5},
                                   {'filter': 64, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 6},
                                   {'filter': 128, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 7}])
     
     # Layer 9 => 11