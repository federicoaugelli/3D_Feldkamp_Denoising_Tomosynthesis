import astra
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import tifffile 
import time 


class Feldkamp:
    def __init__(self, path):
        self.path = path
        self.fp_alg = 'FP3D_CUDA'
        self.fdk_alg = 'FDK_CUDA'

        self.SAD = 6460
        self.SID = 6840
        self.SOD = 6670

        self.scale = 5
        self.detector_pixel_size=0.95
        self.detector_rows = 2000
        self.detector_cols = 3000

        self.detector_x = self.detector_rows*self.detector_pixel_size
        self.detector_y = self.detector_cols*self.detector_pixel_size

        self.angles = np.linspace(-15, 15, 11)

        self.num_projections = len(self.angles)

        self.obj_in_pos=[-1327.46, 1180, -220]
        self.det_in_pos=[-1463.70, 1180, -390]

        self.a = 50
        self.b = 256
        self.c = 256

        self.vol_dim = np.array([self.a, self.b, self.c])

        self.vol_geom, self.proj_geom = self.geom_set_up(self.SAD, self.angles, self.num_projections,
                                                        self.vol_dim, self.det_in_pos, self.obj_in_pos,
                                                        self.detector_pixel_size, self.detector_rows,
                                                        self.detector_cols, self.detector_x, self.detector_y)
        self.volumes = []
        self.metadata = []

    def geom_set_up(self, SAD, angles, num_projections, vol_dim, det_in_pos, obj_in_pos,
                detector_pixel_size, detector_rows, detector_cols, detector_x, detector_y):

        #Inizializziamo la matrice vectors
        vectors=np.ones((num_projections,12)) #matrice di prima che ha dimensione 11x12
    
        #Coordinate della sorgente al variare degli angoli (sorgente ruota attorno ad asse y)
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

        

    def forward_projection(self, vol_id, proj_id, noise_level=0.00):
        print('Inizializzazione dei parametri della FP...\n')
        # Creiamo un dizionario per l'utilizzo della FP
        cfg1 = astra.creators.astra_dict(self.fp_alg)
        cfg1['ProjectionDataId'] = proj_id
        cfg1['VolumeDataId'] = vol_id
        alg1_id = astra.algorithm.create(cfg1)
        
        print('Esecuzione delle proiezione del volume...\n')
        astra.algorithm.run(alg1_id)
    
        # Otteniamo il risultato come un array che avrà le dimensioni del detector
        proj_volume_clean = astra.data3d.get(proj_id)
        print(f'Le dimensioni delle proiezioni sono {np.shape(proj_volume_clean)}\n')
        
        #Aggiunta di rumore alle proiezioni
        if noise_level > 0:
            np.random.seed(10)
            e=np.random.normal(0, 1, np.shape(proj_volume_clean))
            e=e.astype(np.float32)
            e=e/np.linalg.norm(e.flatten(), 2)
            print(f"La norma e è {np.linalg.norm(e.flatten(), 2)}\n")
            proj_volume=proj_volume_clean+e*noise_level*np.linalg.norm(proj_volume_clean.flatten(), 2)
            proj_volume[proj_volume < 0] = 0
        else:
            proj_volume = proj_volume_clean
    
        # # Rappresentiamo alcune proiezioni
        # index = [3, 5, 7, 10]
        # fig = plt.figure(figsize=(20, 5))
        # for i in range(len(index)):
        #     plt.subplot(1, 4, i + 1)
        #     plt.imshow(proj_volume[:, index[i], :], cmap='gray')
        #     plt.title(f'Proiezione angolo {angles[index[i]]}°')
    
        # #Salviamo le proiezioni
        # plt.show()
        #print('Salvataggio delle proiezioni in corso...\n')
        #projection_transposed = np.transpose(proj_volume, (1, 0, 2))  
        # Salva il file come TIFF
        #output_file_reconstructions = os.path.join(path, f"Projections_nl{noise_level}.tiff")
        #tifffile.imwrite(output_file_reconstructions, projection_transposed)
        #print('Salvataggio completato!\n')
        #print('_'*100)

        return proj_volume


    def feldkamp_reconstruction(self, proj_id, vol_id):
        # Ricostruiamo il volume di partenza con la FDK
        print("Inizializzazione dei parametri della FDK...\n")
        reconstruction_id_FDK = astra.data3d.create('-vol', self.vol_geom)
        cfg2 = astra.creators.astra_dict(self.fdk_alg)
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
        slices=[20, 25, 30, 40]
        fig = plt.figure(figsize=(20, 5))
        for i in range(len(slices)):
            plt.subplot(1, 4, i + 1)
            plt.imshow(reconstruction_FDK[slices[i], :, :], cmap='gray', vmin=minvalr, vmax=maxvalr)
            plt.title(f'Ricostruzione slice {slices[i]}\n')
            print('_'*100)
        
        #temp=res(proj_volume, reconstruction_FDK)
        #res_vec_FDK=np.ones((7,1))*temp
        
        #temp=rel_err(phantom, reconstruction_FDK)
        #relerr_vec_FDK=np.ones((7,1))*temp
        
        #temp=PSNR(phantom, reconstruction_FDK)
        #PSNR_vec_FDK=np.ones((7,1))*temp
        
        
        # # Specifiche per salvare l'immagine
        # plt.show()
        #print('Salvataggio delle immagini in corso...\n')
        # Salva il file come TIFF
        #output_file_reconstructions = os.path.join(path, f"Reconstructions_nl{noise_level}.tiff")
        #tifffile.imwrite(output_file_reconstructions, reconstruction_FDK)
        #print('Salvataggio completato!\n')
        #print('_'*100)

        astra.data3d.delete(reconstruction_id_FDK)

        return reconstruction_FDK



    def save_reconstructions(self, filename):
        # Save the volumes and metadata to a file
        volume_name = [m['volume_name'] for m in self.metadata]
        noise_level = [m['noise_level'] for m in self.metadata]
        with open(filename, 'wb') as f:
            np.savez(f, 
                volumes=self.volumes, 
                volume_name=volume_name,
                noise_level=noise_level,
                metadata=self.metadata)
        print(f'Reconstructions saved to {filename}')
 


    def run(self, noise_levels=[0.00, 0.05]):
        # iterate through all the files in the directory
        for volume in os.listdir(self.path):
            if volume.endswith('.tif'):
                phantom = tifffile.imread(os.path.join(self.path, volume))

                for noise_level in noise_levels:
                    vol_id = astra.data3d.create('-vol', self.vol_geom, phantom)
                    proj_id = astra.data3d.create('-proj3d', self.proj_geom)

                    # execute forward projection 
                    proj_volume = self.forward_projection(vol_id=vol_id, proj_id=proj_id, noise_level=noise_level)
                
                    astra.data3d.store(proj_id, proj_volume)

                    reconstruction_FDK = self.feldkamp_reconstruction(proj_id=proj_id, vol_id=vol_id)

                    # Save the reconstruction
                    self.volumes.append(reconstruction_FDK)
                    self.metadata.append({
                        'volume_name': volume,
                        'noise_level': noise_level,
                    })

                    # delete the data to free memory 
                    astra.data3d.delete(vol_id)
                    astra.data3d.delete(proj_id)

        # detele all the algorithms
        #astra.algorithm.delete(alg1_id)
        #astra.algorithm.delete(alg2_id)

        # Save all reconstructions to a file
        self.save_reconstructions('reconstructions.npz')
