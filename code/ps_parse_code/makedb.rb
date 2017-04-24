#!/usr/bin/ruby

# Create a json database filled with goodies


eps_files = Dir.glob("./**/*.eps")

eps_files.each do |file|
  dirs  = file.split("/")
  year  = dirs[1][0,2]
  month = dirs[1][2,4]
  name  = dirs[2]

  if(!name[/\p{L}/].nil?) then
    # name is old arXiv style identifier
    category = name.delete("0-9")
    entry = name.delete("^0-9")

    links = "https://arxiv.org/abs/#{category}/#{entry}"
    ref  = "https://arxiv.org/pdf/#{category}/#{entry}.pdf"

  else
    # name has the new style
    links = "https://arxiv.org/abs/#{name}"
    ref  = "https://arxiv.org/pdf/#{name}.pdf"
  end

  title = dirs[-1]
  title = title.chomp(".eps")
  puts(title)
  author = "W.R. Casper"
  comments = "This JSON database entry was automatically generated using automated algorithms applied to eps images pulled from arXiv"

  puts("reading #{file}")
  output = `./getgraph.py -a \"#{author}\" -r #{ref} -c \"#{comments}\" -t #{title} -n #{name} --links #{links} #{file}`
  puts(output)
  if(File.exists?("#{name}.json")) then
    `mv #{name}.json json/#{name}_#{title}.json`
    `mv #{name}.png  json/#{name}_#{title}.png`
    `mv tmp.ps json/#{name}_#{title}.ps`
  end

end

