import numpy as np
import astra

num_projections=len(angles)

def geometry(vol, det_row, det_col, angles, src_dist, origin_dist):
    vectors=np.ones((num_projections,11))
    vol_geom = astra.create_vol_geom(vol.shape[0], vol.shape[1], vol.shape[2])
    proj_geom = astra.create_proj_geom('cone_vec', det_row, det_col, vectors)

def forward_projection():
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

def feldkamp():
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

