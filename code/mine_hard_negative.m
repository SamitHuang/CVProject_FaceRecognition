function [mine_w, mine_b] =  mine_hard_negative(features_pos, features_neg, non_face_scn_path, w, b, feature_params)

    non_face_scenes = dir( fullfile( non_face_scn_path, '*.jpg' ));

    temp_size = feature_params.template_size;
    cell_size = feature_params.hog_cell_size;

    step_size = cell_size;
    features = [];

    for i = 1:length(non_face_scenes)
          
        fprintf('Mining hard negative in %s\n', non_face_scenes(i).name);
        img = single(imread( fullfile( non_face_scn_path, non_face_scenes(i).name )));
        if(size(img,3) > 1)
        img = rgb2gray(img);
        end
        
        [height, width, ch] = size(img);

        for row = 1 : step_size : height-temp_size
            for col = 1 : step_size : width-temp_size
                img_temp = img(row:row+temp_size-1, col:col+temp_size-1,:);
                hog = vl_hog(img_temp, cell_size);
                feat = hog(:)';
                conf = feat*w + b;
                if conf > 0 
                    features = [features; feat];
                end
            end
        end
        if length(features)>length(features_pos)
            break;
        end
    end

    fprintf('Hard faces number %d\n', length(feat));

    new_feat_neg =  [features; features_neg];

    pos_len = size(features_pos, 1);
    neg_len = size(features_neg, 1);
    label_pos = ones(pos_len, 1);
    label_neg = -1*ones(neg_len, 1);
    label = [label_pos; label_neg];

    feature = [features_pos', new_feat_neg(1:neg_len, :)'];

    [mine_w, mine_b] = vl_svmtrain(feature, label, 1e-4);

    %save('var_mine_w.mat', 'mine_w');
    %save('var_mine_b.mat', 'mine_b');

