from pathlib import Path

def load_dataset_dir(path: str)->Path | None:
    dataset_dir = Path(path)
    
    if not dataset_dir.is_dir():
        print("Pasta Leaves não foi localizada.")
        print("Faça o download do dataset e extraia dentro do diretório principal.")
        
        return None
    
    print("Pasta Leaves localizada.")
    return dataset_dir


def load_img_files_names(folder: Path | None) -> list | None:
    if folder is None:
        print("Não foi possível criar a lista de imagens da pasta Leaves.")
        return

    files_list = [file for file in folder.glob("*.jpg")]

    if len(files_list)==0:

        print("Nenhuma imagem foi localizada na pasta Leaves.")
        print("Verifique se a extração foi executada corretamente.")
        return None
    
    print("Lista de imagens do dataset Flavia Leaf foi criada.")

    return sorted(files_list)

def load_dataset_label_vector():
    labels = {
        "Pubescent Bamboo": [1001,1059],
        "Chinese Horse Chestnut": [1060,1122],
        "Chinese Redbud": [1123,1194],
        "True Indigo": [1195,1267],
        "Japanese Maple": [1268,1323],
        "Nanmu": [1324,1385],
        "Castor Aralia": [1386,1437],
        "Goldenrain Tree": [1438,1496],
        "Chinese Cinnamon": [1497,1551],
        "Anhui Barberry": [1552,1616],
        "Big-fruited Holly": [2001,2050],
        "Japanese Cheesewood": [2051,2113],
        "Wintersweet": [2114,2165],
        "Camphortree": [2166,2230],
        "Japan Arrowwood": [2231,2290],
        "Sweet Osmanthus": [2291,2346],
        "Deodar": [2347,2423],
        "Maidenhair Tree": [2424,2485],
        "Crape Myrtle": [2486,2546],
        "Oleander": [2547,2612],
        "Yew Plum Pine": [2616,2675],
        "Japanese Flowering Cherry": [3001,3055],
        "Glossy Privet": [3056,3110],
        "Chinese Toon": [3111,3175],
        "Peach": [3176,3229],
        "Ford Woodlotus": [3230,3281],
        "Trident Maple": [3282,3334],
        "Beales Barberry": [3335,3389],
        "Southern Magnolia": [3390,3446],
        "Canadian Poplar": [3447,3510],
        "Chinese Tulip Tree": [3511,3563],
        "Tangerine": [3566,3621]
    }

    labels_vetor = []
    labels_encoding = []
    encoding = 0

    for key, values in labels.items():
        labels_encoding.append(key)
        for i in range(values[0],values[1]+1):
            labels_vetor.append(encoding)
        encoding+=1
    
    return labels_vetor,labels_encoding





    

    