require 'fileutils'

#ラベルをつける
# ['kanako','shiori','ayaka','momoka','reni']
labels = {
  'kanako' => '0',
  'shiori' => '1',
  'ayaka' => '2',
  'momoka' => '3',
  'reni' => '4'
}

train_data_path = "./data/train/data.txt"
test_data_path = "./data/test/data.txt"

FileUtils.touch(train_data_path) unless FileTest.exist?(train_data_path)
FileUtils.touch(test_data_path) unless FileTest.exist?(test_data_path)

test_label_rows = []
train_label_rows = []
labels.each do |talent, label|
  test_data_paths = Dir.glob("./data/test/#{talent}/*.jpg")
  train_data_paths = Dir.glob("./data/train/#{talent}/*.jpg")

  test_data_paths.each { |path| test_label_rows.push("#{path} #{label}")}
  train_data_paths.each { |path| train_label_rows.push("#{path} #{label}")}
end

File.open(test_data_path, "w") do |f|
  test_label_rows.each { |row| f.puts(row) }
end

File.open(train_data_path, "w") do |f|
  train_label_rows.each { |row| f.puts(row) }
end
