def normalize(vol):
  return (vol - vol.min()) / (vol.max() - vol.min())

def split_volume(vol, chunk_size=10, overlap=0):
  depth = vol.shape[0]
  chunks = []

  step = chunk_size - overlap

  for start_idx in range(0, depth - chunk_size + 1, step):
    chunk = vol[start_idx:start_idx+chunk_size]
    chunks.append(chunk)

  return chunks
