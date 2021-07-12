import json
import urllib.request
import asyncio
import cv2
import numpy as np

train_img = []
train_mask = []

with open('./school_hallway-test1 (1).json', 'r') as f:
    annotations = json.load(f)  # dataset from segments.ai


async def url_to_image(url):
    resp = await loop.run_in_executor(None, urllib.request.urlopen, url)
    image = np.asarray(bytearray(resp.read()))
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    return cv2.resize(image, (256, 256))


async def pro1(i):
    return await url_to_image(annotations['dataset']['samples'][i]['attributes']['image']['url'])


async def pro2(i):
    return await url_to_image(
        annotations['dataset']['samples'][i]['labels']['ground-truth']['attributes']['segmentation_bitmap']['url'])


async def main_pro():
    global train_img, train_mask
    ti = [asyncio.ensure_future(pro1(i)) for i in range(3000) if
          annotations['dataset']['samples'][i]['labels']['ground-truth']]
    ta = [asyncio.ensure_future(pro2(i)) for i in range(3000) if
          annotations['dataset']['samples'][i]['labels']['ground-truth']]
    train_img = await asyncio.gather(*ti)
    train_mask = await asyncio.gather(*ta)


loop = asyncio.get_event_loop()  # 이벤트 루프를 얻음
loop.run_until_complete(main_pro())  # main이 끝날 때까지 기다림
loop.close()  # 이벤트 루프를 닫음

image_set = []
annotation_set = []
for img in train_img:
  img = cv2.resize(img,(256,256))
  image_set.append(image_set)
for img in train_mask:
  img = cv2.resize(img,(256,256))
  annotation_set.append(image_set)
print(len(image_set),len(annotation_set))

img_s = np.array(image_set)
ano_s = np.array(annotation_set)
np.save('./img.npy', img_s)
np.save('./ano.npy', ano_s)