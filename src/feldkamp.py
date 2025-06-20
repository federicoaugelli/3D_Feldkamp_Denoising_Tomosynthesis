import astra
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import tifffile
import time
from tqdm import tqdm, trange

class Feldkamp:
    def __init__(self, path, angles=(-15, 15), num_projections=11):
        self.path = path
        self.fp_alg = 'FP3D_CUDA'
        self.fdk_alg = 'FDK_CUDA'
        self.SAD = 800
        self.scale = 5
        self.detector_pixel_size = 0.95
        self.detector_rows = 500
        self.detector_cols = 700
        self.detector_x = self.detector_rows*self.detector_pixel_size
        self.detector_y = self.detector_cols*self.detector_pixel_size
        self.angles = np.linspace(angles[0], angles[1], num_projections)
        self.num_projections = len(self.angles)
        self.obj_in_pos = [-128, 128, -25] 
        self.det_in_pos = [-256, 128, -25]
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
        vectors = np.ones((num_projections, 12))
        
        # src cordinates
        rad_angles = angles * (np.pi/180)                        # rad
        vectors[:, 0] = SAD * np.sin(rad_angles)                 # Xs
        vectors[:, 1] = -vol_dim[1]/2                           # Ys
        vectors[:, 2] = SAD * np.cos(rad_angles) + vol_dim[0]/2  # Zs
        
        print("Coordinate della sorgente ottenute:")
        print(f"{'x':>10} {'y':>10} {'z':>10}")
        for row in vectors[:, :3]:
            print(f"{row[0]:>10.2f} {row[1]:>10.2f} {row[2]:>10.2f}")
        
        # detector cordinates (respect to ASTRA)
        Xc = 0                                                   # detector_x/2                                                    
        Yc = 0                                                   # detector_y/2
        Zc = det_in_pos[2]-(vol_dim[2]/2+obj_in_pos[2])          # Pos Z of detector                         
        
        vectors[:, 3] = Xc    # Xc
        vectors[:, 4] = Yc    # Yc
        vectors[:, 5] = Zc    # Zc
        
        u = np.array([detector_pixel_size, 0, 0])/detector_pixel_size  
        v = np.array([0, detector_pixel_size, 0])/detector_pixel_size
        
        # Coordinates u e v
        vectors[:, 6] = u[0]    # Xu
        vectors[:, 7] = u[1]    # Yu
        vectors[:, 8] = u[2]    # Zu
        vectors[:, 9] = v[0]    # Xv
        vectors[:, 10] = v[1]   # Yv
        vectors[:, 11] = v[2]   # Zv
        
        # volume and projection geometries
        vol_geom = astra.create_vol_geom(vol_dim[1], vol_dim[2], vol_dim[0])  # row Y, cols X, slices Z
        proj_geom = astra.create_proj_geom('cone_vec', detector_rows, detector_cols, vectors)  # cone geometry
        
        return [vol_geom, proj_geom]

    def forward_projection(self, vol_id, proj_id, noise_level=0.00, visualize=False):
        with tqdm(total=3, desc="Forward Projection", leave=False) as pbar:
            # Init FP params
            cfg1 = astra.creators.astra_dict(self.fp_alg)
            cfg1['ProjectionDataId'] = proj_id
            cfg1['VolumeDataId'] = vol_id
            alg1_id = astra.algorithm.create(cfg1)
            pbar.update(1)
            pbar.set_description("Esecuzione proiezione")
            
            # FP execution
            astra.algorithm.run(alg1_id)
            pbar.update(1)
            
            proj_volume_clean = astra.data3d.get(proj_id)
            pbar.set_description(f"Proiezioni: {np.shape(proj_volume_clean)}")
            
            # Optional noise
            if noise_level > 0:
                np.random.seed(10)
                e = np.random.normal(0, 1, np.shape(proj_volume_clean))
                e = e.astype(np.float32)
                e = e/np.linalg.norm(e.flatten(), 2)
                proj_volume = proj_volume_clean + e * noise_level * np.linalg.norm(proj_volume_clean.flatten(), 2)
                proj_volume[proj_volume < 0] = 0
                pbar.set_description(f"Rumore aggiunto: {noise_level}")
            else:
                proj_volume = proj_volume_clean
            pbar.update(1)
            
            # Delete memory
            astra.algorithm.delete(alg1_id)

            if visualize:
                index = [0, 5, 7, 10]
                fig = plt.figure(figsize=(20, 5))
                for i in range(len(index)):
                    plt.subplot(1, 4, i + 1)
                    plt.imshow(proj_volume[:, index[i], :], cmap='gray')
                    plt.title(f'Proiezione angolo {self.angles[index[i]]}°')
            
            return proj_volume

    def feldkamp_reconstruction(self, proj_id, vol_id, visualize=False):
        with tqdm(total=3, desc="FDK Reconstruction", leave=False) as pbar:
            # Init FDK params
            reconstruction_id_FDK = astra.data3d.create('-vol', self.vol_geom)
            cfg2 = astra.creators.astra_dict(self.fdk_alg)
            cfg2['ProjectionDataId'] = proj_id
            cfg2['ReconstructionDataId'] = reconstruction_id_FDK
            alg2_id = astra.algorithm.create(cfg2)
            pbar.update(1)
            
            # FDK execution
            pbar.set_description("Esecuzione FDK")
            tstart_FDK = time.time()
            astra.algorithm.run(alg2_id)
            tfinish_FDK = time.time()
            execution_time = tfinish_FDK - tstart_FDK
            pbar.update(1)
            
            reconstruction_FDK = astra.data3d.get(reconstruction_id_FDK)
            pbar.set_description(f"Volume: {np.shape(reconstruction_FDK)}, tempo: {execution_time:.2f}s")
            pbar.update(1)
            
            # Visualize
            if visualize:
                self.visualize_slices(reconstruction_FDK)
            
            # Delete memory
            astra.algorithm.delete(alg2_id)
            astra.data3d.delete(reconstruction_id_FDK)
            
            return reconstruction_FDK
    
    def visualize_slices(self, volume):
        maxvalr = np.max(volume)
        minvalr = np.min(volume)
        slices = [10, 20, 30, 40]
        
        fig = plt.figure(figsize=(20, 5))
        for i in range(len(slices)):
            plt.subplot(1, 4, i + 1)
            plt.imshow(volume[slices[i], :, :], cmap='grey', vmin=minvalr, vmax=maxvalr)
            plt.title(f'Ricostruzione slice {slices[i]}')
        plt.tight_layout()
        plt.show()

    def save_reconstructions(self, filename):
        with tqdm(total=1, desc=f"Salvataggio in {filename}", leave=True) as pbar:
            volume_name = [m['volume_name'] for m in self.metadata]
            noise_level = [m['noise_level'] for m in self.metadata]
            
            with open(filename, 'wb') as f:
                np.savez(f,
                    volumes=np.array(self.volumes),
                    volume_name=volume_name,
                    noise_level=noise_level,
                    metadata=self.metadata)
            pbar.update(1)

    def run(self, noise_levels=[0.00, 0.05], visualize=False):
        volumes_list = [v for v in os.listdir(self.path) if v.endswith('.tif')]
        total_iterations = len(volumes_list) * len(noise_levels)
        
        with tqdm(total=total_iterations, desc="Elaborazione volumi", unit="volume") as pbar:
            for volume in volumes_list:
                # phantom
                phantom = tifffile.imread(os.path.join(self.path, volume))
                
                for noise_level in noise_levels:
                    pbar.set_description(f"Volume: {volume}, Rumore: {noise_level}")
                    
                    vol_id = astra.data3d.create('-vol', self.vol_geom, phantom)
                    proj_id = astra.data3d.create('-proj3d', self.proj_geom)
                    
                    # Forward projection
                    proj_volume = self.forward_projection(vol_id=vol_id, proj_id=proj_id, noise_level=noise_level, visualize=visualize)
                    astra.data3d.store(proj_id, proj_volume)

                    # FDK reconstruction
                    reconstruction_FDK = self.feldkamp_reconstruction(proj_id=proj_id, vol_id=vol_id, visualize=visualize)
                    
                    # Save
                    self.volumes.append(reconstruction_FDK)
                    self.metadata.append({
                        'volume_name': volume,
                        'noise_level': noise_level,
                        'angles': self.angles,
                    })
                    
                    # Delete memory
                    astra.data3d.delete(vol_id)
                    astra.data3d.delete(proj_id)
                    
                    pbar.update(1)
        
        # Save results
        if self.volumes:
            self.save_reconstructions('dataset/reconstruction/reconstructions.npz')
            
            # Stats
            print(f"\nRiepilogo elaborazione:")
            print(f"- Volumi elaborati: {len(volumes_list)}")
            print(f"- Livelli di rumore: {noise_levels}")
            print(f"- Totale ricostruzioni: {len(self.volumes)}")
            print(f"- Dimensione volumi: {self.volumes[0].shape}")
            print(f"- Memoria utilizzata: {np.array(self.volumes).nbytes / (1024**2):.2f} MB")
