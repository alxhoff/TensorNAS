import os
i=0
while 1:
    folder=f'C:\\Users\\mehta\\Desktop\\TensorNAS\\Results\\Result_{i}'
    os.mkdir(folder)
    if os.path.isfile(f'{folder}\\Generations_Visulaization_{i}.eps'):
        i+=1
    else:
        #folder=f'C:\\Users\\mehta\\Desktop\\TensorNAS\\Results\\Result_{i}'

        with open(f'{folder}\\Generations_Visulaization_{i}.txt','w') as f:
            f.write(f'Hello {i}')
        break
