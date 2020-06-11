import os

project_dir_name = os.getcwd()
data_path = os.path.join(project_dir_name, 'Datasets')
training_dir = os.path.join(data_path, 'att_faces', 'Training')
testing_dir = os.path.join(data_path, 'att_faces', 'Testing')

for j in os.walk(testing_dir):
    if len(j[2]):
        for sample_pose in j[2]:
            # if '.pgm' not in sample_pose:
            #     os.remove(j[0] + '/' + sample_pose)
            print(j[0], sample_pose)
            # print(source, destination)
            # shutil.copyfile(source, destination)
            # if 'pose00' in j[0] + '/' + sample_pose:
            #     #     print(dst + '/' + sample)
            #     os.remove(j[0] + '/' + sample_pose)
            # print('############# DONE ' +
            #       sample_pose+'#############')
