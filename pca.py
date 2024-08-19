from os import listdir
from os.path import join, isfile
from eigenvectors import find_eigenvectors, gram_schmidt, found_eigenvectors
from pathlib import Path
import matplotlib.pyplot as plt
import PIL
import numpy as np
from sklearn import svm


IMAGES_PER_PERSON = 3

training_dataset = 'training_set/'
trained_file_pca = 'trained_pca.txt'
trained_file_kpca = 'trained_kpca.txt'
folders = [f for f in listdir(training_dataset)]

print(listdir(training_dataset))

persons = len(folders)
trainings = IMAGES_PER_PERSON * persons

# IMAGE SIZE
width = 480
height = 640
image_area = width * height


def pca_train():
    trained = Path(trained_file_pca)
    if trained.is_file():
        print('Successfully loaded {}'.format(trained_file_pca))
        return np.loadtxt(trained)

    all_faces, person, image_index = parse_faces()
    # Average face. The second arg (0) means that the mean is calculated
    # using all the values at that index from every variant
    mean = np.mean(all_faces, 0)

    # Restando cara media
    all_faces = [all_faces[k, :] - mean for k in range(all_faces.shape[0])]
    # print(all_faces)

    A = np.transpose(all_faces)
    n, m = A.shape
    L = np.dot(all_faces, A)

    L_eigenvectors = find_eigenvectors(A, L)
    C_eigenvectors = np.dot(A, L_eigenvectors)
    for i in range(m):
        C_eigenvectors[:, i] /= np.linalg.norm(C_eigenvectors[:, i])

    np.savetxt(trained_file_pca, C_eigenvectors, fmt='%s')
    return C_eigenvectors


def kpca_train():
    trained = Path(trained_file_kpca)
    if trained.is_file():
        print('Successfully loaded {}'.format(trained_file_kpca))
        return np.loadtxt(trained)

    all_faces, person, image_index = parse_faces()

    degree = 2
    K = (np.dot(all_faces, all_faces.T) / trainings + 1) ** degree
    unoM = np.ones([trainings, trainings]) / trainings
    K = K - np.dot(unoM, K) - np.dot(K, unoM) + np.dot(unoM, np.dot(K, unoM))

    A = np.copy(K)
    m, n = A.shape
    last_R = np.zeros(A.shape)
    eig_vec_K = 1
    found_eigen = False

    while not found_eigen:
        Q, R = gram_schmidt(A)
        A = np.dot(R, Q)
        eig_vec_K = np.dot(eig_vec_K, Q)
        found_eigen = found_eigenvectors(last_R, R)
        last_R = R

    for i in range(m):
        eig_vec_K[:, i] /= np.linalg.norm(eig_vec_K[:, i])

    for col in range(eig_vec_K.shape[1]):
        eig_vec_K[:, col] = eig_vec_K[:, col] / np.sqrt(R[col, col])

    np.savetxt(trained_file_kpca, eig_vec_K, fmt='%s')
    return eig_vec_K



def parse_faces():

    # Matrix with one vector of pixels (len = image_area) per image training
    all_faces = np.zeros([trainings, image_area])
    # Matrix with one vector of pixels (len = image_area) per image training
    person = np.zeros([trainings, 1])
    image_index = 0

    for person_index in range(0, persons):
        person_folder_path = training_dataset + 'person{}/'.format(person_index)
        files = [f for f in listdir(person_folder_path)]
        for file in files:
            # Reading image as a matrix of pixels
            current_path = person_folder_path + file
            img = PIL.Image.open(current_path).convert('L')

            # Converting the matrix to a long array of values between 0 and 1
            img_array = (np.asarray(img.getdata()) / 255)

            # Adding the matrix to the array
            all_faces[image_index, :] = np.reshape(img_array, [1, image_area])
            person[image_index, 0] = person_index
            image_index += 1
        person_index += 1
    return all_faces, person, image_index


def pca_face_input(eigenfaces, input_image):

    all_faces, person, last_index = parse_faces()

    # Test set
    test_number = 1
    test_set = np.zeros([test_number, image_area])
    image_index = 0

    # Adapting new image to match in format with the rest (normalized and with vector shape)
    a = input_image / 255.0
    # Reshape to vector for insertion in 'images'
    test_set[image_index, :] = np.reshape(a, [1, image_area])

    # CARA MEDIA
    # Mean for pixel i using 'images' columns
    mean = np.mean(all_faces, 0)
    test_set = [test_set[k, :] - mean for k in range(test_set.shape[0])]
    max = 100
    # Only first eigen-faces
    B = eigenfaces[:, 0:max]

    # Projecting
    image_projection = np.dot(all_faces, B)
    image_test_projection = np.dot(test_set, B)

    # SVM training
    clf = svm.LinearSVC()
    clf.fit(image_projection, person.ravel())
    labels = clf.predict(image_test_projection)

    return labels[0]


def kpca_face_input(eigenfaces, face_image):
    all_faces, person, last_index = parse_faces()

    #Test set
    test_number = 1
    test_set = np.zeros([test_number,image_area])
    image_index = 0

    # Adapting new image to match in format with the rest (normalized and with vector shape)
    a = face_image / 255.0

    # Reshape to vector for insertion in 'images'
    test_set[image_index, :] = np.reshape(a, [1, image_area])
    
    #KERNEL: polinomial degree
    degree = 2
    K = (np.dot(all_faces, np.transpose(all_faces)) + 1)**degree
            
    #esta transformación es equivalente a centrar las imágenes originales...
    ones = np.ones([trainings, trainings]) / trainings
    K_dot_ones = np.dot(K, ones)
    K = K - np.dot(ones, K) - K_dot_ones + np.dot(ones, K_dot_ones)

    #pre-proyección
    set_pre_proyection = np.dot(np.transpose(K),eigenfaces)
    unoML = np.ones([test_number,trainings])/trainings
    Ktest = (np.dot(test_set,np.transpose(all_faces))/trainings+1)**degree
    ones_dot_K = np.dot(unoML, K)
    Ktest_dot_ones = np.dot(Ktest,ones)
    ones_dot_K_dot_ones = np.dot(unoML,K_dot_ones)
    Ktest = Ktest - ones_dot_K - Ktest_dot_ones + ones_dot_K_dot_ones
    test_set_pre_proyect = np.dot(Ktest,eigenfaces)
    
    nmax = eigenfaces.shape[1]
    nmax = 100
    set_proyection = set_pre_proyection[:,0:nmax]
    test_set_proyection = test_set_pre_proyect[:,0:nmax]
            
    #SVM
    clf = svm.LinearSVC()
    clf.fit(set_proyection,person.ravel())
    labels = clf.predict(test_set_proyection)

    return labels[0]


def show_average_face(av_face):
    resized = np.asarray([f for f in av_face]).reshape(640, 480)*255
    plt.imshow(resized, cmap="gray")
    plt.show()