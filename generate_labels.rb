require 'fileutils'

#ラベルをつける
# ['takemoto','taniai','suzuki']
labels = {
  'takemoto' => '0',
  'taniai' => '1',
  'suzuki' => '2'
}

train_data_path = "./data/train/data.txt"
test_data_path = "./data/test/data.txt"

FileUtils.touch(train_data_path) unless FileTest.exist?(train_data_path)
FileUtils.touch(test_data_path) unless FileTest.exist?(test_data_path)

test_label_rows = []
train_label_rows = []
labels.each do |talent, label|
  test_data_paths = Dir.glob("./data/test/#{talent}/*.png")
  train_data_paths = Dir.glob("./data/train/#{talent}/*.png")

  test_data_paths.each { |path| test_label_rows.push("#{path} #{label}")}
  train_data_paths.each { |path| train_label_rows.push("#{path} #{label}")}
end

File.open(test_data_path, "w") do |f|
  test_label_rows.each { |row| f.puts(row) }
end

File.open(train_data_path, "w") do |f|
  train_label_rows.each { |row| f.puts(row) }
end
