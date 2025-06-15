import astra
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import tifffile 
import time 

"""**1. Creazione volume e geometria di proiezione**

In questa fase non viene calcolato nulla, ma vengono create la geometria del volume e la geometria del proiettore.

Per ogni proiezione costruiamo una matrice vectors=(Xs|Ys|Zs|Xc|Yc|Zc|Xu|Yu|Zu|Xv|Yv|Zv) di 11 righe (come il numero di proiezioni), dove, per ogni proiezione (ruota attorno all'asse y):

   * s=(Xs,Ys,Zs) è la posizione della sorgente ad ogni tempo
   * c=(Xc,Yc,Zc) è la posizione del centro del detector
   * u=(Xu,Yu,Zu) è il vettore che indica nel piano del detector la punta del 1° versore u, rispetto a Oxyz tale che ||u-c||=1
   * v=(Xv,Yv,Zv) è il vettore che indica nel piano del detector la punta del 2° versore v, rispetto a Oxyz tale che ||v-c||=1

Siccome il detector nel nostro caso è fisso, le ultime 9 colonne vengono calcolate una volta e poi rimangono uguali in tutte le proiezioni successive.

Il volume viene centrato nell'origine del sistema Oxyz, quindi le coordinate del detector devono essere calcolate di conseguenza.

Assumo che il centro delgi assi cartesiani SU ASTRA abbia le stesse coordinate di quello dell'IMS, a parte la x'

"""

def geom_set_up(SAD, angles, num_projections, vol_dim, det_in_pos, obj_in_pos,
                detector_pixel_size, detector_rows, detector_cols, detector_x, detector_y):

  #Inizializziamo la matrice vectors
    vectors=np.ones((num_projections,12)) #matrice di prima che ha dimensione 11x12
    
  #Coordinate della sorgente al variare degli angoli
    rad_angles=angles*(np.pi/180)                               #angoli in radianti
    vectors[:,0]=SAD*np.sin(rad_angles)                         #Xs
    vectors[:,1]=-vol_dim[1]/2                                  #Ys
    vectors[:,2]=SAD*np.cos(rad_angles)+vol_dim[0]/2            #Zs
    
  #Stampiamo i valori ottenuti tramite la visualizzazione di una tabella
    print("Le coordinate della sorgente ottenute sono: \n")
    print(f"{'x':>10} {'y':>10} {'z':>10}")

    for row in vectors[:,:3]:
        print(f"{row[0]:>10.2f} {row[1]:>10.2f} {row[2]:>10.2f}")
    
  #Coordinate del centro del detector (rispetto al sistema di riferimento di ASTRA)
    Xc=0                                                                      #detector_x/2                                                    
    Yc=0                                                                      #detector_y/2
    Zc=det_in_pos[2]-(vol_dim[2]/2+obj_in_pos[2])                             #det_in_pos[2]-((vol_dim[2]+obj_in_pos[2])/2) Ricavato con Elena Morotti                           
    
    vectors[:,3]=Xc                   #Xc
    vectors[:,4]=Yc                   #Yc
    vectors[:,5]=Zc                   #Zc

  #Costruiamo i versori ortonormali u e v
    u=np.array([detector_pixel_size, 0, 0])/detector_pixel_size  
    v=np.array([0, detector_pixel_size ,0])/detector_pixel_size
    
  #Coordinate dei versori u e v
    vectors[:,6]=u[0]                    #Xu
    vectors[:,7]=u[1]                    #Yu
    vectors[:,8]=u[2]                    #Zu
    vectors[:,9]=v[0]                    #Xv
    vectors[:,10]=v[1]                   #Yv
    vectors[:,11]=v[2]                   #Zv

  #Creiamo la geometria del volume e della proiezione
    vol_geom=astra.create_vol_geom( vol_dim[1], vol_dim[2], vol_dim[0])  #righe sono le Y, colonne sono le X, Z sono le slices
    #vol_geom=astra.create_vol_geom( vol_dim[2], vol_dim[0], vol_dim[1])
    proj_geom = astra.create_proj_geom('cone_vec', detector_rows, detector_cols, vectors) #geometria a cono con detector di queste dimensioni e la matrice vector come fatta in precedenza

    return [vol_geom, proj_geom]


"""## **2. Allocazione di memoria (su ASTRA) per volume e proiettore**

In questa fase creiamo le id del volume dell'oggetto e del proiettore che ci serviranno poi per usare gli algoritmi.

Il volume e le proiezioni posso essere sia inizializzati con delle immagini reali/volumi proiezioni creati da noi, sia lasciati vuoti (cioè definiti come array di zeri).

Ad esempio, con dati reali non si passa niente come inizializer per il volume, perché viene ricostruito in seguito, mentre le proiezioni verranno caricate e date come initializer.


Il formato dell'eventuale array initializer deve essere convertito in float32.

"""
SAD=6460 #mm
SID=6840 #mm
SOD=6670 #mm

scale=5
detector_pixel_size=0.95             #0.085 #mm
detector_rows=2000                  #pixel
detector_cols=3000                   #pixel

detector_x=detector_rows*detector_pixel_size  #mm 
detector_y=detector_cols*detector_pixel_size  #mm  

#angles=np.array([14.5, 11.6, 8.7, 5.8, 2.9, 0, -2.9, -5.8, -8.7, -11.6, -14.5])            #range=29
angles=np.array([14.5, 11.6, 8.7, 5.8, 2.9, 0, -2.9, -5.8, -8.7, -11.6, -14.5])            #range=29
#angles=np.array([7.5, 6. , 4.5, 3.0 , 1.5,  0.0, -1.5, -3.0, -4.5, -6.0, -7.5])           #range=15
num_projections=len(angles)

#Coordinate di punti rispetto a sdr di IMS
obj_in_pos=[-1327.46, 1180, -220]  #centro dell'oggetto
det_in_pos=[-1463.70, 1180, -390]  #centro del detector

#Dimensioni del volume
#a=1000
#b=2000
#c=350
a=70
b=200
c=400
vol_dim=np.array([a, b, c])
#vol_dim=np.array([b, c, a])

vol_geom, proj_geom=geom_set_up(SAD, angles, num_projections, vol_dim, det_in_pos, obj_in_pos,
                detector_pixel_size, detector_rows, detector_cols, detector_x, detector_y)

#Selezioniamo il volume da utilizzare per proiezione e ricostruzioni
vol_hole=1

if vol_hole==1:
    #Generiamo un parallelepipedo con una cavità per fare simulazioni
    phantom = np.zeros((a, b, c), dtype=np.float32)
    #phantom=np.ones([b, c, a])

    # Definizione delle dimensioni della cavità
    cavity_start = 200/scale
    cavity_end = 800/scale

    # Creazione del parallelepipedo con cavità e con apertura
    # phantom[100 : -100, 100 : -100, 100 : -100] = 0.5
    # phantom [:,  200 : 600, 300 : 400] = 0
    # phantom[160 :210, 700 : 735, 100 : -100] = 2               #filo orizzontale
    # phantom[178 : 183,  100 : -100, 1250 : 1255] = 1           #filo verticale

    phantom[20 : -20, 20 : -20, 20 : -20] = 0.5
    phantom [:,  40 : 120, 60 : 80] = 0
    phantom[32 :42, 140 : 147, 20 : -20] = 2               #filo orizzontale
    phantom[36 : 37,  20 : -20, 250 : 251] = 1  
    #Passiamo il cubo creato sopra come volume iniziale per inizializzare la memoria
    vol_id=astra.data3d.create('-vol', vol_geom, phantom)

    #Inizializziamo la memoria per le proiezioni
    proj_id = astra.data3d.create('-proj3d', proj_geom)

# else:
#     os.chdir(".")
    
#     phantom = tifffile.imread('Rotated_Phantom_big_ImageJ.tif')

#     #Passiamolo come volume iniziale per inizializzare la memoria
#     vol_id=astra.data3d.create('-vol', vol_geom, phantom)

#     #Inizializziamo la memoria per le proiezioni
#     proj_id = astra.data3d.create('-proj3d', proj_geom)

"""
VISUALIZZAZI0NE DEL VOLUME SIMULATO PER ALCUNE SLICES

"""

path="images"   #Percorso dove salviamo le immagini
save_phantom=0

# slices=[20, 105, 180, 245]
# fig=plt.figure(figsize=(20,5))
# for i in range (len(slices)):
#   plt.subplot(1,4,i+1)
#   plt.imshow(phantom[slices[i], :, :], cmap='gray', vmin=0, vmax=1)
#   plt.title(f'Slice {slices[i]}')
# plt.show()
print('ciao')

# if save_phantom==1:
#     output_file_phantom=os.path.join(path, "phantom.tiff")
#     tifffile.imwrite(output_file_phantom, phantom)
# """
# VISUALIZZAZIONE DETECTOR E VETTORI u, v
# """

# # Posizioni e dimensioni del detector
# det_center = np.array([0, 0, det_in_pos[2]-(vol_dim[2]/2+obj_in_pos[2])])                                   # Centro del detector
# u_vector = (np.array([detector_pixel_size, 0, 0])/detector_pixel_size)*15                                 # Vettore u (orizzontale)
# v_vector = (np.array([0, detector_pixel_size, 0])/detector_pixel_size)*15                                 # Vettore v (verticale)

# # Definisci i 4 vertici del detector (rettangolo)                                 
# #ottengo i vertici come somma di vettori e le colonne sono maggiori delle righe 
# corner1 = det_center - u_vector * detector_cols / 2 - v_vector * detector_rows / 2
# corner2 = det_center + u_vector * detector_cols / 2 - v_vector * detector_rows / 2
# corner3 = det_center + u_vector * detector_cols / 2 + v_vector * detector_rows / 2
# corner4 = det_center - u_vector * detector_cols / 2 + v_vector * detector_rows / 2

# # Lista dei vertici per disegnare il detector
# vertices = np.array([corner1, corner2, corner3, corner4, corner1])/100


# # Crea il grafico 3D
# fig = plt.figure(figsize=(10, 10))
# ax = fig.add_subplot(111, projection='3d')

# # Disegna il detector
# ax.plot(vertices[:, 0], vertices[:, 1], vertices[:, 2], label='Detector', color='blue')

# # Disegna il vettore u
# ax.quiver(det_center[0], det_center[1], det_center[2], 
#           10 * u_vector[0],  u_vector[1], 10 * u_vector[2], color='r', length=0.1, label='versore u')

# # Disegna il vettore v
# ax.quiver(det_center[0], det_center[1], det_center[2], 
#           10 * v_vector[0], 10 * v_vector[1], 10 * v_vector[2], color='g', length=0.1, label='versore v')


# ax.set_xlim([-250, 250])
# ax.set_ylim([-200, 200])
# ax.set_zlim([-2000, 2000])

# ax.scatter(det_center[0], det_center[1], det_center[2], s=75, label='Centro del detector', color='blue')

# # Configura il grafico
# ax.set_xlabel('X (mm)')
# ax.set_ylabel('Y (mm)')
# ax.set_zlabel('Z (mm)')
# ax.legend()
# plt.title("Vista dall'alto del detector")

# # Imposta vista dall'alto
# ax.view_init(elev=90, azim=0)

# plt.show()

# """
# VISUALIZZAZIONE POSIZIONE DELLE SORGENTI
# """

# # Calcolo delle posizioni della sorgente
# rad_angles = angles * (np.pi / 180)  # Angoli in radianti
# pos_sorgente = np.zeros((len(angles), 3))  
# pos_sorgente[:, 0] = SAD * np.sin(rad_angles)                            # Coordinate X della sorgente
# pos_sorgente[:, 1] = -vol_dim[1] / 2                                     # Coordinate Y della sorgente
# pos_sorgente[:, 2] = SAD * np.cos(rad_angles)+vol_dim[2]/2               # Coordinate Z della sorgente

# # Definizione del detector 
# det_center = np.array([detector_x / 2, detector_y / 2, det_in_pos[2]-((vol_dim[2]+obj_in_pos[2])/2)])  # Centro del detector
# u_vector = np.array([detector_pixel_size, 0, 0])  # Vettore u (orizzontale)
# v_vector = np.array([0, detector_pixel_size, 0])  # Vettore v (verticale)

# # Definizione dei 4 vertici del detector
# corner1 = det_center - u_vector * detector_cols / 2 - v_vector * detector_rows / 2
# corner2 = det_center + u_vector * detector_cols / 2 - v_vector * detector_rows / 2
# corner3 = det_center + u_vector * detector_cols / 2 + v_vector * detector_rows / 2
# corner4 = det_center - u_vector * detector_cols / 2 + v_vector * detector_rows / 2

# # Lista dei vertici per disegnare il detector
# vertices = np.array([corner1, corner2, corner3, corner4, corner1])

# # Creazione del grafico 3D
# fig = plt.figure(figsize=(10, 10))
# ax = fig.add_subplot(111, projection='3d')

# # Disegno del detector
# ax.plot(vertices[:, 0], vertices[:, 1], vertices[:, 2], label='Detector', color='blue')

# # Disegno delle posizioni della sorgente
# ax.scatter(pos_sorgente[:, 0], pos_sorgente[:, 1], pos_sorgente[:, 2], color='orange', label='Posizioni della Sorgente')

# # Configuriamo il grafico
# ax.set_xlabel('X (mm)')
# ax.set_ylabel('Y (mm)')
# ax.set_zlabel('Z (mm)')

# ax.set_xlim([np.min(pos_sorgente[:, 0]), np.max(det_center[0])])
# ax.set_ylim([np.min(pos_sorgente[:, 1]), np.max(det_center[1])])
# ax.set_zlim([np.min(pos_sorgente[:, 2]), np.max(det_center[2])])

# # Aggiungiamo la legenda e il titolo
# ax.legend()
# plt.title("Posizioni della Sorgente e Detector")

# ax.view_init(elev=90, azim=0)

# # Mostriamo il grafico
# plt.show()

"""
FUNZIONI PER GLI ERRORI
"""
def rel_err(x_true, x_corr):
    # Flatten the images
    x_true = x_true.flatten()
    x_corr = x_corr.flatten()

    # Compute the error
    RE = np.linalg.norm(x_true - x_corr, 2) / np.linalg.norm(x_true, 2)
    return RE

def PSNR(x_true, x_corr):
    # Flatten the images
    x_true = x_true.flatten()
    x_corr = x_corr.flatten()

    mse = np.mean((x_true - x_corr) ** 2)
    if (
        mse == 0
    ):  # MSE is zero means no noise is present in the signal. Therefore PSNR have no importance.
        return 100
    max_pixel = 1
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr

def res(proj_true, x_solutions):
    """Calcoliamo il residuo tra le proiezioni del volume originale (proj_true) e le proiezioni del volume ricostruito con un algoritmo. 
    Per farlo prendiamo in input le proiezioni del volume originale (proj_true) e le ricostruzioni del volume (x_solutions), di cui calcoremo le proiezioni all'interno della funzione (proj_res).
    Matematicamente il residuo sarebbe ||(FP)*volume_originale-proiezioni_volume_ricostruito||
    """
    volres_id=astra.data3d.create('-vol', vol_geom, x_solutions)   #inizializziamo in memoria un volume che abbia le dimensioni di x_solutions (per la FDK saranno le ricostruzioni)
    projres_id = astra.data3d.create('-proj3d', proj_geom)         #inizializziamo in memoria un spazio che sarà occupato dalle proiezioni
    
    cfg5 = astra.creators.astra_dict('FP3D_CUDA')
    cfg5['ProjectionDataId'] = projres_id
    cfg5['VolumeDataId'] = volres_id
    alg5_id = astra.algorithm.create(cfg5)
    print('Esecuzione delle proiezione del volume...\n')
    astra.algorithm.run(alg5_id)

    # Otteniamo il risultato come un array che avrà le dimensioni del detector  
    projres=astra.data3d.get(projres_id)                            #proiezioni di x_solutions
    plt.imshow(proj_volume[:, 5, :], cmap='gray')
    plt.title(f'Proiezione residuo')
    
    # Flatten the images
    p_true = proj_true.flatten()                                    
    p_corrupted = projres.flatten()
    
    # Compute the error
    RES = np.linalg.norm(p_true - p_corrupted, 2)
    return RES

alg1 = 'FP3D_CUDA'
alg2 = 'FDK_CUDA'


#Controlli per eseguire gli algoritmi
execute_FP=1                        
execute_FDK=1


#Controlli per salvare i file tiff
save_projections=1
save_reconstructions_FDK=1

PLOTS=1


#CODICE PER LA FORWRAD PROJECTION (FP)
if  execute_FP == 1:
    print('Inizializzazione dei parametri della FP...\n')
    # Creiamo un dizionario per l'utilizzo della FP
    cfg1 = astra.creators.astra_dict(alg1)
    cfg1['ProjectionDataId'] = proj_id
    cfg1['VolumeDataId'] = vol_id
    alg1_id = astra.algorithm.create(cfg1)
    
    print('Esecuzione delle proiezione del volume...\n')
    astra.algorithm.run(alg1_id)

    # Otteniamo il risultato come un array che avrà le dimensioni del detector
    proj_volume_clean = astra.data3d.get(proj_id)
    print(f'Le dimensioni delle proiezioni sono {np.shape(proj_volume_clean)}\n')
    
    #Aggiunta di rumore alle proiezioni
    noise_level=0.05
    np.random.seed(10)
    e=np.random.normal(0, 1, np.shape(proj_volume_clean))
    e=e.astype(np.float32)
    e=e/np.linalg.norm(e.flatten(), 2)
    print(f"La norma e è {np.linalg.norm(e.flatten(), 2)}\n")
    proj_volume=proj_volume_clean+e*noise_level*np.linalg.norm(proj_volume_clean.flatten(), 2)
    proj_volume[proj_volume < 0] = 0

    # # Rappresentiamo alcune proiezioni
    # index = [3, 5, 7, 10]
    # fig = plt.figure(figsize=(20, 5))
    # for i in range(len(index)):
    #     plt.subplot(1, 4, i + 1)
    #     plt.imshow(proj_volume[:, index[i], :], cmap='gray')
    #     plt.title(f'Proiezione angolo {angles[index[i]]}°')

    # #Salviamo le proiezioni
    # plt.show()
    if save_projections==1:
        print('Salvataggio delle proiezioni in corso...\n')
        projection_transposed = np.transpose(proj_volume, (1, 0, 2))  
        # Salva il file come TIFF
        output_file_reconstructions = os.path.join(path, f"Projections_nl{noise_level}.tiff")
        tifffile.imwrite(output_file_reconstructions, projection_transposed)
        print('Salvataggio completato!\n')
        print('_'*100)
#CODICE PER LA FDK
if execute_FDK == 1:
    # Ricostruiamo il volume di partenza con la FDK
    print("Inizializzazione dei parametri della FDK...\n")
    reconstruction_id_FDK = astra.data3d.create('-vol', vol_geom)
    cfg2 = astra.creators.astra_dict(alg2)
    cfg2['ProjectionDataId'] = proj_id
    cfg2['ReconstructionDataId'] = reconstruction_id_FDK
    alg2_id = astra.algorithm.create(cfg2)
    
    print("Esecuzione della FDK..\n")
    tstart_FDK=time.time()
    astra.algorithm.run(alg2_id)
    tfinish_FDK=time.time()
    time_FDK=tfinish_FDK-tstart_FDK
    time_FDK=np.ones((7,1))*time_FDK

    # Otteniamo il risultato come un array
    reconstruction_FDK = astra.data3d.get(reconstruction_id_FDK)
    print(f'Le dimensioni del volume ricostruito sono {np.shape(reconstruction_FDK)}\n')

    #Calcoliamo i valori massimi e minimi dell'array
    #reconstruction_FDK[reconstruction_FDK < 0] = 0
    maxvalr = np.max(reconstruction_FDK)
    minvalr = np.min(reconstruction_FDK)

    # Visualizziamo alcune slices
    # slices=[20, 105, 180, 245]
    # fig = plt.figure(figsize=(20, 5))
    # for i in range(len(slices)):
    #     plt.subplot(1, 4, i + 1)
    #     plt.imshow(reconstruction_FDK[slices[i], :, :], cmap='gray', vmin=minvalr, vmax=maxvalr)
    #     plt.title(f'Ricostruzione slice {slices[i]}\n')
    #     print('_'*100)
    
    temp=res(proj_volume, reconstruction_FDK)
    res_vec_FDK=np.ones((7,1))*temp
    
    temp=rel_err(phantom, reconstruction_FDK)
    relerr_vec_FDK=np.ones((7,1))*temp
    
    temp=PSNR(phantom, reconstruction_FDK)
    PSNR_vec_FDK=np.ones((7,1))*temp
    
    
    # # Specifiche per salvare l'immagine
    # plt.show()
    if save_reconstructions_FDK==1:
        print('Salvataggio delle immagini in corso...\n')
        # Salva il file come TIFF
        output_file_reconstructions = os.path.join(path, f"Reconstructions_nl{noise_level}.tiff")
        tifffile.imwrite(output_file_reconstructions, reconstruction_FDK)
        print('Salvataggio completato!\n')
        print('_'*100)


# aggiungere il plot dei residui, degli errori relativi e del PSNR.