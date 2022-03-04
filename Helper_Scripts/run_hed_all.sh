for file in ../../Data/1obj/*;do
  for img in $file/src_color/*;do
    python3 detect_edges_image.py -i "$img" -d hed_model -s "$file"
  done
done
