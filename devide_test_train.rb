require 'fileutils'

talents = ['takemoto','taniai','suzuki']

train_image_dir = './data/train/'
test_image_dir = './data/test/'

#以下、学習データとテストデータにわける
RATIO_TRAIN_DATA = 0.7

talents.each do |dir|
  FileUtils.mkdir_p(train_image_dir+dir) unless FileTest.exist?(train_image_dir+dir)
  FileUtils.mkdir_p(test_image_dir+dir) unless FileTest.exist?(test_image_dir+dir)

  sum = Dir.glob("./member_images/cropped_images_#{dir}/*.png").count
  puts "sum of #{dir}:#{sum.to_s}"

  Dir.glob("./member_images/cropped_images_#{dir}/*.png").each_with_index do |src, index|
    percent = index.to_f/sum.to_f
    if percent < RATIO_TRAIN_DATA
      FileUtils.cp(src, train_image_dir+dir+"/")
    else
      FileUtils.cp(src, test_image_dir+dir+"/")
    end
  end

  puts "train:"
  puts Dir.glob("./#{train_image_dir+dir}/*.png").count
  puts "test:"
  puts Dir.glob("./#{test_image_dir+dir}/*.png").count
end
