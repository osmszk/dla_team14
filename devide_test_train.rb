require 'fileutils'

#ももくろに戻すときはtrue
is_momokuro = false

talents = is_momokuro ? ['kanako','shiori','ayaka','momoka','reni'] : ['takemoto','taniai','suzuki']
ext = is_momokuro ? 'jpg' : 'png'

train_image_dir = './data/train/'
test_image_dir = './data/test/'

#以下、学習データとテストデータにわける
RATIO_TRAIN_DATA = 0.7

talents.each do |dir|
  FileUtils.mkdir_p(train_image_dir+dir) unless FileTest.exist?(train_image_dir+dir)
  FileUtils.mkdir_p(test_image_dir+dir) unless FileTest.exist?(test_image_dir+dir)

  sum = Dir.glob("./image_momo/cropped_images_#{dir}/*.#{ext}").count
  puts "sum of #{dir}:#{sum.to_s}"

  Dir.glob("./image_momo/cropped_images_#{dir}/*.#{ext}").each_with_index do |src, index|
    percent = index.to_f/sum.to_f
    if percent < RATIO_TRAIN_DATA
      FileUtils.cp(src, train_image_dir+dir+"/")
    else
      FileUtils.cp(src, test_image_dir+dir+"/")
    end
  end

  puts "train:"
  puts Dir.glob("./#{train_image_dir+dir}/*.#{ext}").count
  puts "test:"
  puts Dir.glob("./#{test_image_dir+dir}/*.#{ext}").count
end
