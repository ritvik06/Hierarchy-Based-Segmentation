for file in ../../Data/1obj/*;do
	for img in $file/src_color/*;do
		python slic.py -i "$img" -d "$file"
	done
done
