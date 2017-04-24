#!/usr/bin/ruby

##
# Program to search the arXiv for files containing certain strings
##

src_dir = "/home/wcasper/arXiv/src"

tar_files = `ls #{src_dir}/*.tar`.split()

search_words = ["graph","edge","vertex","morphism"]

tar_files.each do |tf|
  # extract the tar file
  puts("Extracting #{tf}")
  `tar -xf #{tf}`
  date = tf.split("_")[2]
  # unzip individual files and search for text
  gzip_files = `ls #{date}/*.gz`.split()
  gzip_files.each do |gzf|
    puts("unzipping #{gzf}")
    out_name = gzf.chomp(".gz")
    `gunzip #{gzf}`
    filetype = `file #{out_name}`
    match = false
    if filetype.include?("tar") then
      `mv #{out_name} #{out_name}.tar`
      Dir.mkdir(out_name) if not Dir.exists?(out_name)
      `tar -xf #{out_name}.tar -C #{out_name}`
      tex_file = Dir.glob("#{out_name}/**/*.{tex,ltx,TEX,LTX}")[0]
      if(not tex_file == nil and tex_file.length() > 4) then
        puts("latex file is #{tex_file}")
        match = true
        search_words.each do |word|
          result = `cat -v #{tex_file} | grep -i #{word}`
          if not result.include?(word) then
            match = false
            break
          end
        end
      end
    end
    if match then
      puts("got something in #{tex_file}")
    else
      `rm -rf #{out_name}`
    end
    `rm #{gzf}` if File.exists?(gzf)
  end
  `rm #{date}/*.pdf`
  `rm #{date}/*.tar`
end
puts(tar_files)

