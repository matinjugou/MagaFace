import cv2

from face_match_sdk.face_matcher import FaceMatcher

if __name__ == "__main__":
    image_path = 'data/test.jpg'
    img = cv2.imread(image_path)

    face_matcher = FaceMatcher()

    # execute detect
    faces = face_matcher.extract_faces(img)

    # print
    print(f'There are {len(faces)} faces in {image_path}')

    for i in range(len(faces) - 1):
        for j in range(i + 1, len(faces)):
            score, likely = face_matcher.compare_faces(face1=faces[i], face2=faces[j])
            print(f'face {i + 1} and face{j + 1} {"is" if likely else "is not"} the same person. (score = {score})')

    # visualize
    show_image = True
    if show_image:
        img = face_matcher.visualize(img, faces)
        cv2.imshow("", img)
        cv2.waitKey(0)
