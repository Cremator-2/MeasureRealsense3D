import pyrealsense2 as rs
import numpy as np
import cv2
import math


def max_size_of_object(img, final_mask, depth, scale, min_area, floor_level):
    # ф-ция определения габаритных размеров

    contours, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # находим контура на маске

    dist_max = 0.0
    width = 0
    height = 0
    x1 = 0
    y1 = 0

    for cnt in contours:  # цикл обработки контуров

        area = cv2.contourArea(cnt)  # определение площади пятна
        if area > min_area:  # если пятно меньше заданного - игнорируем его (позволяет отфильтровать шумы и артефакты)
            rect = cv2.minAreaRect(cnt)  # находим минимальный прямоугольник, который описывает деталь
            box = cv2.boxPoints(rect)  # получаем координаты его углов
            box = np.int0(box)

            # нахождение угла поворота
            # вычисление координат двух векторов, являющихся сторонам прямоугольника
            edge1 = np.int0((box[1][0] - box[0][0], box[1][1] - box[0][1]))
            edge2 = np.int0((box[2][0] - box[1][0], box[2][1] - box[1][1]))

            # выясняем какой вектор больше
            used_edge = edge1
            if cv2.norm(edge2) > cv2.norm(edge1):
                used_edge = edge2
            reference = (1, 0)  # вектор, задающий горизонт
            # вычисляем угол между самой длинной стороной прямоугольника и горизонтом
            angle = 180.0 / math.pi * math.acos(
                (reference[0] * used_edge[0] + reference[1] * used_edge[1]) /
                (cv2.norm(reference) * cv2.norm(used_edge)))

            cv2.drawContours(img, [box], 0, (255, 0, 0), 2)  # рисуем прямоугольник
            (xt, yt, wt, ht) = cv2.boundingRect(cnt)  # находим прямоугольную область кадра, в которой находиться объект
            cv2.rectangle(img, (xt, yt), (xt + wt, yt + ht), (0, 255, 255), 2)  # выделяем ее прямоугольником
            # выводим в кадр величину угла наклона
            cv2.putText(img, "%d" % int(angle), (xt + wt + 10, yt - 10), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (255, 0, 0), 2)

            z_height_arr_max = depth[yt:(yt + ht), xt:(xt + wt)]
            # сопоставляем найденной области, область карты глубины
            z_height_max = np.min(z_height_arr_max[np.nonzero(z_height_arr_max)])
            # находим минимальное ненулевое значение, что является самой высокой точкой объекта
            ind = np.where(z_height_arr_max == z_height_max)  # находим координаты данной точки
            # масштабируем, получая дистанцию до нее в мм
            dist_max = z_height_arr_max[ind[0][0]][ind[1][0]] * scale * 1000

            x1 = ind[1][0] + xt  # находим координаты точки в полном кадре
            y1 = ind[0][0] + yt

            cv2.circle(img, (x1, y1), 3, (0, 0, 255), 2)  # рисуем круг в самой высокой точке объекта

            fg_masked = cv2.bitwise_and(depth, depth, mask=final_mask)  # срезаем с карты глубин пол
            kernel = np.ones((3, 3), np.uint8)  # параметры пикселизации карты глубин
            fg_masked = cv2.erode(fg_masked, kernel, iterations=4)  # пикселизируем карту глубины для того, чтобы
            # получить более равномерное измерение высоты, избавиться от флуктуаций

            z_height_arr_min = fg_masked[yt:(yt + ht), xt:(xt + wt)]  # находим самую низкую точку объекта
            z_height_min = np.max(z_height_arr_min[np.nonzero(z_height_arr_min)])  # аналогично самой высокой
            ind1 = np.where(z_height_arr_min == z_height_min)
            dist_min = z_height_arr_min[ind1[0][0]][ind1[1][0]] * scale * 1000

            x2 = ind1[1][0] + xt
            y2 = ind1[0][0] + yt

            cv2.circle(img, (x2, y2), 3, (255, 255, 0), 2)

            k = 0.00000000113278 * floor_level ** 2 - 0.00000289238293 * floor_level + 0.00307494503798
            b = -0.00000084846447 * floor_level ** 2 + 0.00223034705269 * floor_level - 1.17497382144753

            f_x = k * dist_min + b  # считаем плотность пикселей в зависимости от минимальной точки объекта по высоте

            width = f_x * rect[1][0]  # высчитываем максимальную ширину
            height = f_x * rect[1][1]  # высчитываем максимальную длину

    if dist_max == 0:  # выводим "None" если нет объекта в кадре, в этом случае возвращаются нули
        cv2.putText(img, 'None', (10 + x1, 30 + y1), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 0, 255), 1)

    return [img, dist_max, width, height]


def init_l515(clipping_distance_in_mm):
    # ф-ия инициализации лидара

    data_pipeline = rs.pipeline()
    config = rs.config()

    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    profile = data_pipeline.start(config)

    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale_coefficient = depth_sensor.get_depth_scale()  # получение данных о масштабировании карты глубины

    floor_clipping_distance = (clipping_distance_in_mm / 1000) / depth_scale_coefficient
    align_to = rs.stream.color  # сглаживание
    align_frames = rs.align(align_to)
    return floor_clipping_distance, depth_scale_coefficient, align_frames, data_pipeline


def create_mask(img):
    # ф-ця, которая возвращает маску из всех цветов, кроме черного

    l_h = 0
    l_s = 0
    l_v = 1
    u_h = 180
    u_s = 255
    u_v = 255

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    lower = np.array([l_h, l_s, l_v])
    upper = np.array([u_h, u_s, u_v])

    mask = cv2.inRange(hsv, lower, upper)
    return mask


def get_pre_mask(clipping_distance_level, color_image_frame, depth_image_frame):
    # функция "среза" поверхности пола с цветного изображения
    # результатом ф-ции является цветное изображение, где поверхность пола черная,
    # а объекты выше уровня пола видны в кадре
    # это необходимо для того, чтобы создать маску кадра, которая не будет учитывать фон,
    # что позволит выделить интересующие объекты в независимости от их цвета, оттенка, освещения

    bg_color = 0
    depth_image_3d = np.dstack(
        (depth_image_frame, depth_image_frame, depth_image_frame))  # depth image is 1 channel, color is 3 channels
    color_image_frame = np.where((depth_image_3d > clipping_distance_level) | (depth_image_3d <= 0),
                                 bg_color, color_image_frame)
    return color_image_frame


if __name__ == '__main__':

    floor_dist = 905  # [mm]
    amendment = 20  # [mm]
    floor = floor_dist - amendment

    q1 = 0
    q2 = 480
    p1 = 0
    p2 = 640

    clipping_distance, depth_scale, align, pipeline = init_l515(floor)

    while True:

        frames = pipeline.wait_for_frames()
        frames = align.process(frames)

        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()

        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())

        color_image = color_image[q1:q2, p1:p2]
        depth_image = depth_image[q1:q2, p1:p2]

        color_image = get_pre_mask(clipping_distance,
                                   color_image,
                                   depth_image)

        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs
                                           (depth_image, alpha=0.05),
                                           cv2.COLORMAP_HOT)

        binary_image = create_mask(color_image)

        [color_image, z, x, y] = max_size_of_object(color_image,
                                                    binary_image,
                                                    depth_image,
                                                    depth_scale,
                                                    1000,
                                                    floor_dist)

        text = 'Size: ' + str(int(y)) + 'x' + str(int(x)) + 'x' + str(int(floor_dist - z)) + ' mm'
        if x != 0:
            cv2.putText(color_image,
                        text, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 0, 255), 1)

        unused_field = np.full((q2 - q1, p2 - p1, 3), 240, np.uint8)

        mask_3d = np.dstack((binary_image, binary_image, binary_image))

        cv2.imshow('3D Measuring Intel RealSense L515', np.vstack([
            np.hstack([depth_colormap, mask_3d]),
            np.hstack([color_image, unused_field])
        ]))

        key = cv2.waitKey(10)
        if key == 27:
            cv2.destroyAllWindows()
            break

    pipeline.stop()
