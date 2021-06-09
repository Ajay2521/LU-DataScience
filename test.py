def img_to_encoding(image_path, model):

    # loading the image with resizing the image
    img = load_img(image_path, target_size=(160, 160))
    print('\nImage data :',img)
    print("\nImage Data :",img.shape)

    # converting the img data in the form of pixcel values
    img = np.around(np.array(img) / 255.0, decimals=12)
    print('\nImage pixcel value :',img)
    print("\nImage pixcel shape :",img.shape)

    # expanding the dimension for making the image to fit for the input
    x_train = np.expand_dims(img, axis=0)
    print('\nx_train :',x_train)
    print("\nx_train shape :",x_train.shape)

    # predicting the embedding of the image by passing the image to the model
    embedding = model.predict(x_train)
    print('\nEmbedding :',embedding)
    print('\nEmbedding shape :',embedding.shape)

    # linalg.norm - used to calculate one of the eight different matrix norms or vector norms
    # linalg - used to return the eigenvalues and eigenvectors of a real symmetric matrix.
    embedding = embedding / np.linalg.norm(embedding, ord=2) 
    print('\nEmbedding :',embedding)
    print('\nEmbedding shape :',embedding.shape)

    return embedding

# importing the database in the program as a dict datatype
face_database = {}

# listdir - used to get the list of all files and directories in the specified directory.
for folder_name in os.listdir('database'):
  print('\nfolder_name :',folder_name)
  # path.join - concatenates various path components with exactly one directory separator ('/')
	
  for image_name in os.listdir(os.path.join('database',folder_name)): # database/foldder_name
	
  	print('\nimage_name :',image_name)
   
    # splitext - used to split the path name into a pair root and extension.
    # basename - used to get the base name in specified path
		user_name = os.path.splitext(os.path.basename(image_name))
		print('\nUser name : \n',user_name)
		
    # img_to_encoding - used to get the face embedding for a image
    face_database[user_name] = img_to_encoding(os.path.join('database',folder_name,image_name), model)
 
print(face_database)

