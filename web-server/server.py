# sanic==23.3.0
from typing import Dict, List, Any

import numpy as np
import torch
from PIL import Image
from sanic import Sanic
from sanic.response import text
from predictor_service import Predictor
from sanic.response import json
from sanic.exceptions import NotFound

from segment_anything.utils.amg import mask_to_rle_pytorch
from web_tool import ServerException,ErrorCode
from io import BytesIO
import json as bejson
from config import SEGMENT_OBJ,MODEL_PATH

app = Sanic("SegmentAnythingWebServer")
model_server = Predictor(model_path=MODEL_PATH)

@app.get("/")
async def ping(request):
    print(request)
    return text("pong")
@app.exception(NotFound)
async def ignore_404s(request, exception):
    return text('403', status=403)

@app.exception(ServerException)
async def ignore_404s(request, exception: ServerException):
    return json({"data": None, "status": exception.status, "msg": exception.message})

# 成功以及失败的返回脚本
def ok(data):
    return json({"data": data, "status": 0, "msg":""})

def fail(data, httpCode=500):
    return json({"data": data, "status": httpCode}, httpCode)

@app.route('/segment', methods=['POST'])
def segment(request):
    # 获取上传的图像文件和关键点
    image_file = request.files.get('image').body
    params: Dict = None
    if 'data' in request.form:
        # 检查参数是否是 json
        params_text = request.form['data'][0]
        if not isDictStr(params_text):
            return fail('form data must be Json String', 412)
        params = bejson.loads(params_text)
        print(params)

    # 读取图像数据并进行处理
    image = Image.open(BytesIO(image_file)).convert('RGB')
    image_array = np.array(image)
    w, h, _ = image_array.shape

    input_boxs = None
    input_points = None
    input_labels = None
    if params is None:
        masks = model_server.generate_auto(image_array)
    else:
        points = params.get("points")
        labels = params.get("labels")
        box = params.get("box")
        if box is not None and points is not None:
            raise ServerException(ErrorCode.PARAM_ERROR)
        if labels is None and points is not None:
            # 生成 labels 数据
            pass
        if points is not None:
            input_points = np.array(points)
            num = input_points.shape[0]
            if labels is None:
                input_labels = np.ones(num, dtype=np.int16)
            else:
                if input_points.shape[0] != num:
                    raise ServerException(ErrorCode.PARAM_NOT_REQUIRED, "labels")
                input_labels = np.array(labels)
        else:
            input_boxs = np.array(box)

        masks, scores = model_server.generate(image_array
                                      , input_points=input_points
                                      , input_labels=input_labels
                                      , input_boxs=input_boxs
                                      )
        # from pycocotools import mask as mask_utils  # type: ignore
        # mask_list = []
        # for i in range(masks.shape[0]):
        #     rle = mask_utils.encode(np.asfortranarray(masks[i]))
        #     mask_list.append(rle)
        #     from segment_anything.utils.amg import mask_to_rle_pytorch
        #     a = mask_to_rle_pytorch(torch.from_numpy(masks[i]))
        rles = mask_to_rle_pytorch(torch.from_numpy(masks))
        result = []
        scores = scores.tolist()
        for i in range(len(scores)):
            item = SEGMENT_OBJ.copy()
            item["segmentation"] = rles[i]
            item["stability_score"] = scores[i]
            item["crop_box"] = [0, 0, h, w]
            item["point_coords"] = points
            item["bbox"] = box
            result.append(item)
        return ok(result)
    return ok(masks)



# 判断一个字符串是否可以转化为字典
def isDictStr(String):
    try:
        bejson.loads(String)
        return True
    except:
        return False

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5015, workers=4)
