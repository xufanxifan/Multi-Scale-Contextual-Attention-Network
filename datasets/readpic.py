from PIL import Image
import numpy
import os

# def main():
#     # image_path = "/home/user/data/facadewhu/facadewhu/facadewhu_sort/data/labelledfile/SegmentationClassPNG/val/Paris/paris_2017_10_dXBvj5PC2M-zoKOt0d5lWg.png"  # 读取图片
#     # image_path = "/home/user/data/hkscapesplus_gai/data/labelledfile/SegmentationClasslabelIds/test/hongkong/Track_B-CAM1-4_2018.11.19_02.36.55(610)labelIds.png"
#     labelled_root_path = "/home/user/data/facadewhu/facadewhu/facadewhu_sort/data/labelledfile"
#
#     for (root, dirs, files) in os.walk(labelled_root_path):
#         for file in files:
#             if file.endswith('png'):
#                 image_path=os.path.join(root,file)
#                 save_path_1=image_path.replace('SegmentationClassPNG', 'SegmentationClasslabelIds')
#                 save_path=save_path_1.replace('.png', 'labelIds.png')
#                 if os.path.exists(save_path):
#                     continue
#                 image = Image.open(image_path)
#                 re_img = numpy.asarray(image)
#                 newpic = numpy.empty([re_img.shape[0], re_img.shape[1]], dtype=int)
#                 for x in range(0, re_img.shape[0]):
#                     for y in range(0, re_img.shape[1]):
#                         if re_img[x][y][0] == 0 and re_img[x][y][1] == 0 and re_img[x][y][2] == 0:
#                             newpic[x][y] = 0
#                         elif re_img[x][y][0] == 255 and re_img[x][y][1] == 0 and re_img[x][y][2] == 0:
#                             newpic[x][y] = 1
#                         elif re_img[x][y][0] == 255 and re_img[x][y][1] == 128  and re_img[x][y][2] == 0:
#                             newpic[x][y] = 2
#                         elif re_img[x][y][0] == 255 and re_img[x][y][1] == 255 and re_img[x][y][2] == 0:
#                             newpic[x][y] = 3
#                         elif re_img[x][y][0] == 128 and re_img[x][y][1] == 0 and re_img[x][y][2] == 255:
#                             newpic[x][y] = 4
#                         elif re_img[x][y][0] == 0 and re_img[x][y][1] == 0 and re_img[x][y][2] == 255:
#                             newpic[x][y] = 5
#                         elif re_img[x][y][0] == 0 and re_img[x][y][1] == 255 and re_img[x][y][2] == 0:
#                             newpic[x][y] = 6
#                 newpicpic=Image.fromarray(numpy.uint8(newpic))
#                 # newpicpic.show()
#                 newpicpic.save(save_path)
#                 print(file)
#
#     test = 1


def main():
    # image_path = "/home/user/data/facadewhu/facadewhu/facadewhu_sort/data/labelledfile/SegmentationClassPNG/val/Paris/paris_2017_10_dXBvj5PC2M-zoKOt0d5lWg.png"  # 读取图片
    # image_path = "/home/user/data/hkscapesplus_gai/data/labelledfile/SegmentationClasslabelIds/test/hongkong/Track_B-CAM1-4_2018.11.19_02.36.55(610)labelIds.png"
    labelled_root_path = "/home/user/data/facadewhu/facadewhu/facadewhu_resize/data"

    for (root, dirs, files) in os.walk(labelled_root_path):
        for file in files:
            if file.endswith('jpg'):
                image_path=os.path.join(root,file)
                image = Image.open(image_path)
                image_1=image.resize([2046,2046],resample=Image.NEAREST)
                image_1.save(image_path)
                print(file)

    test = 1



if __name__ == '__main__':
    main()
