import os

project_dir_name = os.getcwd()
data_path = os.path.join(project_dir_name, 'Datasets')
training_dir = os.path.join(data_path, 'att_faces', 'Training')
testing_dir = os.path.join(data_path, 'att_faces', 'Testing')

for j in os.walk(testing_dir):
    if len(j[2]):
        for filename in j[2]:
            if '.DS_Store' in filename:
                os.remove(j[0] + '/' + filename)
                print(j[0], filename)
